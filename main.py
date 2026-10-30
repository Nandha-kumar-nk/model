"""
================================================================================
SMART GARBAGE CLASSIFICATION SYSTEM - LIVE CAMERA & ESP32 CONTROL
================================================================================
Core Application: OpenCV Live Stream + Real AI Model + PySerial Hardware Link
Three Categories:
  1. ORGANIC   -> "ORGANIC"
  2. INORGANIC -> "INORGANIC"
  3. REUSABLE  -> "REUSABLE"
================================================================================
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import cv2

# Optional serial import with fallback safety
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# Import TensorFlow / TFLite
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ------------------------------------------------------------------------------
# GLOBAL CONFIGURATION & SYSTEM PARAMETERS
# ------------------------------------------------------------------------------
MODEL_PATH = "models/waste_classifier.tflite"
FALLBACK_KERAS_PATH = "models/waste_classifier.keras"

# 4 AI Classes (Exact Index Mapping)
CLASS_NAMES = ["ORGANIC", "INORGANIC", "REUSABLE"]

# Centralized ESP32 Hardware Command Mapping
BIN_COMMANDS = {
    "ORGANIC": "ORGANIC",
    "INORGANIC": "INORGANIC",
    "REUSABLE": "REUSABLE"
}

# Color Theme for Visual HUD (BGR Format for OpenCV)
CLASS_COLORS = {
    "ORGANIC": (46, 204, 113),         # Vibrant Green
    "INORGANIC": (52, 73, 94),
    "REUSABLE": (243, 156, 18)
}

# UI Palette
COLOR_BG_PANEL = (24, 28, 36)
COLOR_TEXT_WHITE = (245, 245, 245)
COLOR_TEXT_DIM = (160, 165, 175)
COLOR_ACCENT_GREEN = (46, 204, 113)
COLOR_ACCENT_YELLOW = (41, 211, 255)
COLOR_ACCENT_RED = (60, 60, 235)

# Default Algorithm Thresholds
DEFAULT_CONFIDENCE_THRESHOLD = 0.70  # 70% threshold
DEFAULT_CONSECUTIVE_FRAMES = 5       # 5 stable frames required for confirmation
DEFAULT_COMMAND_COOLDOWN = 5.0       # 5 seconds wait after opening bin
DEFAULT_SERIAL_PORT = "COM3"
DEFAULT_BAUD_RATE = 115200


# ==============================================================================
# ROBUST CAMERA INITIALIZATION SYSTEM (WINDOWS CAP_DSHOW & CAP_MSMF SUPPORT)
# ==============================================================================
def initialize_camera(preferred_index=None, max_index_to_test=3):
    """
    Robust camera initialization that tests camera indices 0, 1, 2 using
    Windows-native backends (cv2.CAP_DSHOW and cv2.CAP_MSMF).
    
    Returns:
        cap (cv2.VideoCapture or None)
        active_index (int or None)
        active_backend (str or None)
    """
    print("=" * 75)
    print(" [CAMERA SUBSYSTEM] INITIALIZING LIVE VIDEO FEED")
    print("=" * 75)

    backends_to_try = [
        ("DirectShow (CAP_DSHOW)", cv2.CAP_DSHOW),
        ("Media Foundation (CAP_MSMF)", cv2.CAP_MSMF),
        ("Default OpenCV Backend", None)
    ]

    indices_to_test = [preferred_index] if preferred_index is not None else list(range(max_index_to_test))

    for index in indices_to_test:
        print(f"\n[SCAN] Testing Camera Index {index}...")

        for backend_name, backend_flag in backends_to_try:
            print(f"  -> Probing Index {index} with {backend_name}...", end=" ", flush=True)
            try:
                if backend_flag is not None:
                    cap = cv2.VideoCapture(index, backend_flag)
                else:
                    cap = cv2.VideoCapture(index)

                if cap is not None and cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        print("SUCCESS! (Frame stream active)")
                        print("=" * 75)
                        print(f" [CAMERA CONNECTED] Camera Index: {index} | Backend: {backend_name}")
                        print("=" * 75)

                        # Set default 720p resolution
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        return cap, index, backend_name
                    else:
                        print("FAILED (Device opened but frame was empty)")
                        cap.release()
                else:
                    print("FAILED (Device could not be opened)")
                    if cap is not None:
                        cap.release()
            except Exception as e:
                print(f"FAILED (Exception: {e})")

    # Diagnostic error printout if no camera found
    print("\n" + "!" * 75)
    print(" [ERROR] CAMERA NOT FOUND / CAMERA ACCESS FAILED")
    print("!" * 75)
    print(" Possible reasons on Windows laptops:")
    print("   1. Camera privacy permissions are OFF:")
    print("      -> Open Windows Settings > Privacy & security > Camera")
    print("      -> Enable 'Camera access' and 'Let desktop apps access your camera'")
    print("   2. Another application is currently locking the webcam (Teams, Zoom, Browser tabs)")
    print("   3. Webcam hardware switch or privacy shutter is closed")
    print("   4. Webcam driver error in Device Manager")
    print("!" * 75 + "\n")
    return None, None, None


# ==============================================================================
# ESP32 SERIAL HARDWARE CONTROLLER
# ==============================================================================
class ESP32Controller:
    """
    Manages serial communication with the ESP32-S3 microcontroller.
    Features non-blocking connection, error isolation, and demo mode.
    """
    def __init__(self, port=DEFAULT_SERIAL_PORT, baud_rate=DEFAULT_BAUD_RATE, demo_mode=False):
        self.port = port
        self.baud_rate = baud_rate
        self.demo_mode = demo_mode
        self.serial_conn = None
        self.is_connected = False
        self.last_command = "NONE"
        self.last_command_time = 0

        if not self.demo_mode:
            self.connect()

    def connect(self):
        """Attempts connection to ESP32 without crashing if disconnected."""
        if not SERIAL_AVAILABLE:
            print("[SERIAL WARNING] 'pyserial' not installed. Switching to DEMO HARDWARE MODE.")
            self.demo_mode = True
            self.is_connected = False
            return False

        try:
            print(f"[SERIAL] Attempting connection to ESP32 on {self.port} at {self.baud_rate} baud...")
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=0.1,
                write_timeout=0.5
            )
            time.sleep(1.5)  # Allow ESP32 boot cycle / DTR reset
            self.is_connected = True
            print(f"[SERIAL SUCCESS] Connected to ESP32-S3 on {self.port}!")
            return True
        except Exception as e:
            self.is_connected = False
            print(f"[SERIAL WARNING] Could not connect to ESP32 on {self.port}: {e}")
            print(f"[SERIAL INFO] Operating in DEMO HARDWARE MODE or Standalone AI Mode.")
            return False

    def send_bin_command(self, category_name):
        """
        Transmits the specified bin command to ESP32 or displays simulated output.
        """
        command_str = BIN_COMMANDS.get(category_name, category_name.upper())
        self.last_command = command_str
        self.last_command_time = time.time()

        if self.demo_mode:
            print(f"\n[HARDWARE SIMULATION] >>> SIMULATED COMMAND: {command_str} <<<")
            return True, f"SIMULATED: {command_str}"

        if not self.is_connected or self.serial_conn is None:
            # Try lazy re-connect
            if not self.connect():
                print(f"[HARDWARE ERROR] ESP32 NOT CONNECTED. Command '{command_str}' not dispatched.")
                return False, "ESP32 NOT CONNECTED"

        try:
            payload = f"{command_str}\n".encode('utf-8')
            self.serial_conn.write(payload)
            self.serial_conn.flush()
            print(f"\n[SERIAL TX -> ESP32] >>> COMMAND SENT: '{command_str}\\n' <<<")
            return True, f"SENT: {command_str}"
        except Exception as e:
            print(f"[SERIAL ERROR] Failed sending command to ESP32: {e}")
            self.is_connected = False
            return False, "SERIAL TX FAILED"

    def read_response(self):
        """Non-blocking read for incoming ESP32 telemetry/ACK."""
        if self.is_connected and self.serial_conn and self.serial_conn.in_waiting > 0:
            try:
                line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    print(f"[SERIAL RX <- ESP32] {line}")
                return line
            except Exception:
                pass
        return None

    def close(self):
        """Safely closes the serial connection."""
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.close()
                print("[SERIAL] Connection closed safely.")
            except Exception:
                pass


# ==============================================================================
# AI MODEL INFERENCE ENGINE (TFLITE & KERAS SUPPORT)
# ==============================================================================
class AIClassifier:
    """
    Inference engine supporting TensorFlow Lite (.tflite) and full Keras (.keras) models.
    """
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.is_tflite = False
        self.interpreter = None
        self.keras_model = None
        self.input_shape = (224, 224)
        self.input_dtype = np.float32
        self.input_index = None
        self.output_index = None
        self.scale = 1.0
        self.zero_point = 0

        self.load_model()

    def load_model(self):
        target_path = self.model_path
        if not os.path.exists(target_path) and os.path.exists(FALLBACK_KERAS_PATH):
            print(f"[MODEL] '{target_path}' not found, falling back to '{FALLBACK_KERAS_PATH}'")
            target_path = FALLBACK_KERAS_PATH

        if not os.path.exists(target_path):
            print("\n" + "!" * 75)
            print(" [ERROR] MODEL FILE REQUIRED")
            print("!" * 75)
            print(f" Trained model was not found at: '{self.model_path}'")
            print(" Please train the model and convert it to TFLite by running:")
            print("   1. python train_model.py")
            print("   2. python convert_to_tflite.py")
            print(" Or place your pre-trained model file in the 'models/' directory:")
            print("   models/waste_classifier.tflite")
            print("!" * 75 + "\n")
            raise FileNotFoundError(f"MODEL FILE REQUIRED: {self.model_path}")

        print(f"[MODEL] Loading model from: {target_path}")

        if target_path.endswith(".tflite"):
            self.is_tflite = True
            self.interpreter = tf.lite.Interpreter(model_path=target_path)
            self.interpreter.allocate_tensors()

            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()

            self.input_index = input_details[0]['index']
            self.output_index = output_details[0]['index']
            shape = input_details[0]['shape']
            self.input_shape = (shape[1], shape[2])
            self.input_dtype = input_details[0]['dtype']

            if 'quantization' in input_details[0]:
                self.scale, self.zero_point = input_details[0]['quantization']
                if self.scale == 0:
                    self.scale = 1.0

            print(f"[MODEL] TFLite Engine Initialized | Input: {self.input_shape} | Dtype: {self.input_dtype}")
        else:
            self.is_tflite = False
            self.keras_model = tf.keras.models.load_model(target_path)
            input_shape = self.keras_model.input_shape
            self.input_shape = (input_shape[1], input_shape[2])
            print(f"[MODEL] Keras Engine Initialized | Input: {self.input_shape}")

    def preprocess_frame(self, frame_bgr):
        # 1. Resize to input shape
        resized = cv2.resize(frame_bgr, self.input_shape, interpolation=cv2.INTER_LINEAR)
        # 2. Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # 3. Expand batch dimension [1, H, W, 3]
        tensor = np.expand_dims(rgb, axis=0)

        # 4. Format dtype
        if self.is_tflite:
            if self.input_dtype == np.int8:
                tensor = (tensor / self.scale + self.zero_point).astype(np.int8)
            else:
                tensor = tensor.astype(np.float32)
        else:
            tensor = tensor.astype(np.float32)

        return tensor

    def predict(self, frame_bgr):
        input_tensor = self.preprocess_frame(frame_bgr)

        if self.is_tflite:
            self.interpreter.set_tensor(self.input_index, input_tensor)
            self.interpreter.invoke()
            raw_output = self.interpreter.get_tensor(self.output_index)[0]
            output_details = self.interpreter.get_output_details()
            if output_details[0]['dtype'] == np.int8:
                out_scale, out_zp = output_details[0]['quantization']
                probs = (raw_output.astype(np.float32) - out_zp) * out_scale
            else:
                probs = raw_output.astype(np.float32)
        else:
            probs = self.keras_model.predict(input_tensor, verbose=0)[0]

        # Softmax normalization safety
        if np.sum(probs) > 0 and (np.max(probs) > 1.0 or np.min(probs) < 0.0):
            exp_p = np.exp(probs - np.max(probs))
            probs = exp_p / np.sum(exp_p)

        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])
        pred_class = CLASS_NAMES[top_idx]

        return pred_class, confidence, probs


# ==============================================================================
# OPENCV HEAD-UP DISPLAY (HUD)
# ==============================================================================
class HUDOverlay:
    @staticmethod
    def draw_glass_panel(img, pt1, pt2, color=COLOR_BG_PANEL, alpha=0.78):
        overlay = img.copy()
        cv2.rectangle(overlay, pt1, pt2, color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv2.rectangle(img, pt1, pt2, (60, 68, 80), 1)

    @staticmethod
    def draw_hud(frame, state):
        h, w = frame.shape[:2]

        # 1. TOP HEADER BANNER
        HUDOverlay.draw_glass_panel(frame, (0, 0), (w, 55), (15, 18, 24), alpha=0.90)
        cv2.putText(frame, "SMART GARBAGE CLASSIFICATION SYSTEM", (20, 34),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, COLOR_TEXT_WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {state['fps']:.1f}", (w - 110, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_ACCENT_YELLOW, 1, cv2.LINE_AA)

        # 2. LEFT SIDE PANEL: PROBABILITIES
        panel_w = 260
        panel_h = 180
        HUDOverlay.draw_glass_panel(frame, (15, 70), (15 + panel_w, 70 + panel_h), COLOR_BG_PANEL, alpha=0.82)
        cv2.putText(frame, "CLASS PROBABILITIES", (25, 92),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        probs = state['probs']
        bar_start_y = 112
        bar_max_w = 130

        for i, (c_name, prob) in enumerate(zip(CLASS_NAMES, probs)):
            y = bar_start_y + (i * 30)
            color = CLASS_COLORS.get(c_name, (200, 200, 200))
            is_winner = (c_name == state['pred_class']) and (state['confidence'] >= state['conf_thresh'])

            label_color = COLOR_TEXT_WHITE if is_winner else COLOR_TEXT_DIM
            cv2.putText(frame, f"{c_name[:7]}", (25, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, label_color, 1 if not is_winner else 2, cv2.LINE_AA)

            cv2.rectangle(frame, (90, y), (90 + bar_max_w, y + 12), (40, 45, 55), -1)
            fill_w = int(bar_max_w * np.clip(prob, 0.0, 1.0))
            if fill_w > 0:
                cv2.rectangle(frame, (90, y), (90 + fill_w, y + 12), color, -1)

            pct_str = f"{prob * 100:4.1f}%"
            cv2.putText(frame, pct_str, (90 + bar_max_w + 8, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

        # 3. RIGHT SIDE PANEL: PREDICTION CARD
        card_w = 320
        card_h = 180
        card_x1 = w - card_w - 15
        card_y1 = 70
        HUDOverlay.draw_glass_panel(frame, (card_x1, card_y1), (w - 15, card_y1 + card_h), COLOR_BG_PANEL, alpha=0.85)

        cv2.putText(frame, "TARGET CLASSIFICATION", (card_x1 + 15, card_y1 + 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        pred_class = state['pred_class']
        conf = state['confidence']
        is_high_conf = conf >= state['conf_thresh']

        if is_high_conf:
            primary_color = CLASS_COLORS.get(pred_class, COLOR_ACCENT_GREEN)
            status_text = "HIGH CONFIDENCE"
            display_pred = pred_class.replace("_", " ")
        else:
            primary_color = (80, 80, 240)
            status_text = "LOW CONFIDENCE"
            display_pred = "UNCERTAIN"

        cv2.putText(frame, display_pred, (card_x1 + 15, card_y1 + 55),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, primary_color, 2, cv2.LINE_AA)

        conf_str = f"Confidence: {conf * 100:.1f}% (Req: {state['conf_thresh']*100:.0f}%)"
        cv2.putText(frame, conf_str, (card_x1 + 15, card_y1 + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

        streak = state['stability_streak']
        req_streak = state['required_frames']
        streak_str = f"Stability: {streak}/{req_streak} frames"
        cv2.putText(frame, streak_str, (card_x1 + 15, card_y1 + 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        for b in range(req_streak):
            bx = card_x1 + 15 + (b * 22)
            by = card_y1 + 115
            b_color = primary_color if b < streak else (50, 55, 65)
            cv2.rectangle(frame, (bx, by), (bx + 16, by + 8), b_color, -1)

        rec_bin = f"Bin: {pred_class.replace('_', ' ')} BIN" if is_high_conf else "Bin: [STANDBY]"
        cv2.putText(frame, rec_bin, (card_x1 + 15, card_y1 + 148),
                    cv2.FONT_HERSHEY_DUPLEX, 0.50, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

        cv2.putText(frame, f"[{status_text}]", (card_x1 + 15, card_y1 + 168),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, primary_color, 1, cv2.LINE_AA)

        # 4. CENTER RETICLE
        box_size = 200
        cx, cy = w // 2, h // 2
        bx1, by1 = cx - box_size // 2, cy - box_size // 2
        bx2, by2 = cx + box_size // 2, cy + box_size // 2
        reticle_color = primary_color if is_high_conf else (100, 110, 120)

        corner_len = 25
        cv2.line(frame, (bx1, by1), (bx1 + corner_len, by1), reticle_color, 2)
        cv2.line(frame, (bx1, by1), (bx1, by1 + corner_len), reticle_color, 2)
        cv2.line(frame, (bx2, by1), (bx2 - corner_len, by1), reticle_color, 2)
        cv2.line(frame, (bx2, by1), (bx2, by1 + corner_len), reticle_color, 2)
        cv2.line(frame, (bx1, by2), (bx1 + corner_len, by2), reticle_color, 2)
        cv2.line(frame, (bx1, by2), (bx1, by2 - corner_len), reticle_color, 2)
        cv2.line(frame, (bx2, by2), (bx2 - corner_len, by2), reticle_color, 2)
        cv2.line(frame, (bx2, by2), (bx2, by2 - corner_len), reticle_color, 2)

        # 5. COOLDOWN / LOW CONFIDENCE BANNER
        if state['in_cooldown']:
            remain = max(0.0, state['cooldown_end'] - time.time())
            HUDOverlay.draw_glass_panel(frame, (cx - 220, cy - 25), (cx + 220, cy + 25), (10, 25, 10), alpha=0.9)
            cooldown_msg = f"BIN DISPATCHED -> WAITING FOR NEXT ITEM ({remain:.1f}s)"
            cv2.putText(frame, cooldown_msg, (cx - 200, cy + 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.48, COLOR_ACCENT_GREEN, 1, cv2.LINE_AA)
        elif not is_high_conf:
            HUDOverlay.draw_glass_panel(frame, (cx - 210, cy + box_size // 2 + 10), (cx + 210, cy + box_size // 2 + 45), (20, 20, 45), alpha=0.85)
            cv2.putText(frame, "LOW CONFIDENCE - PLEASE SHOW WASTE CLEARLY", (cx - 195, cy + box_size // 2 + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (90, 160, 255), 1, cv2.LINE_AA)

        # 6. FOOTER
        footer_h = 55
        HUDOverlay.draw_glass_panel(frame, (0, h - footer_h), (w, h), (15, 18, 24), alpha=0.92)

        cam_status = f"[CAM: INDEX {state['cam_idx']} ({state['cam_backend'].split(' ')[0]})]"
        cv2.putText(frame, cam_status, (20, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        if state['demo_mode']:
            hw_badge = "[MODE: DEMO (SIMULATION)]"
            hw_color = COLOR_ACCENT_YELLOW
        elif state['esp32_connected']:
            hw_badge = f"[ESP32: CONNECTED ({state['serial_port']})]"
            hw_color = COLOR_ACCENT_GREEN
        else:
            hw_badge = f"[ESP32: DISCONNECTED ({state['serial_port']})]"
            hw_color = COLOR_ACCENT_RED

        cv2.putText(frame, hw_badge, (230, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.44, hw_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Last: {state['last_action_msg']}", (20, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        controls_str = "[Q/ESC]: Exit | [R]: Reset | [D]: Demo | [C]: Reconnect Cam"
        cv2.putText(frame, controls_str, (w - 480, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)


# ==============================================================================
# MAIN APPLICATION CONTROLLER
# ==============================================================================
def run_garbage_classifier(
    model_path=MODEL_PATH,
    preferred_camera=None,
    conf_thresh=DEFAULT_CONFIDENCE_THRESHOLD,
    required_frames=DEFAULT_CONSECUTIVE_FRAMES,
    cooldown_sec=DEFAULT_COMMAND_COOLDOWN,
    serial_port=DEFAULT_SERIAL_PORT,
    baud_rate=DEFAULT_BAUD_RATE,
    demo_mode=False
):
    print("=" * 80)
    print(" SMART GARBAGE CLASSIFICATION SYSTEM - INITIALIZING")
    print("=" * 80)

    # 1. Initialize Camera First
    cap, cam_idx, cam_backend = initialize_camera(preferred_index=preferred_camera)
    if cap is None:
        print("[FATAL ERROR] Webcam initialization failed. Exiting.")
        sys.exit(1)

    # 2. Initialize AI Model
    ai_engine = AIClassifier(model_path=model_path)

    # 3. Initialize ESP32 Hardware Link
    esp32 = ESP32Controller(port=serial_port, baud_rate=baud_rate, demo_mode=demo_mode)

    # State variables
    current_candidate_class = None
    stability_streak = 0
    in_cooldown = False
    cooldown_end_time = 0
    last_action_msg = "Awaiting waste item..."

    stats = {
        "total_dispatched": 0,
        "counts": {"ORGANIC": 0, "INORGANIC": 0, "REUSABLE": 0},
        "session_start": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    fps = 0.0
    frame_count = 0
    fps_start_time = time.time()

    print("\n" + "=" * 80)
    print(" [SYSTEM READY] Real-Time Waste Classifier Running!")
    print(" Controls:")
    print("   'Q' or 'ESC' -> Exit")
    print("   'R'          -> Reset Stability Tracker")
    print("   'D'          -> Toggle Demo Mode")
    print("   'C'          -> Reconnect Camera")
    print("=" * 80 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Frame dropped. Attempting quick camera re-read...")
                time.sleep(0.03)
                continue

            frame_count += 1
            if time.time() - fps_start_time >= 1.0:
                fps = frame_count / (time.time() - fps_start_time)
                frame_count = 0
                fps_start_time = time.time()

            esp32.read_response()

            current_time = time.time()
            if in_cooldown and (current_time >= cooldown_end_time):
                in_cooldown = False
                stability_streak = 0
                current_candidate_class = None
                last_action_msg = "Ready for next item"

            # AI Inference
            pred_class, confidence, probs = ai_engine.predict(frame)

            # Decision Engine
            if not in_cooldown:
                if confidence >= conf_thresh:
                    if pred_class == current_candidate_class:
                        stability_streak += 1
                    else:
                        current_candidate_class = pred_class
                        stability_streak = 1

                    if stability_streak >= required_frames:
                        confirmed_class = current_candidate_class
                        target_bin = BIN_COMMANDS[confirmed_class]
                        
                        print("\n" + "#" * 70)
                        print(f" [CONFIRMED] >>> {confirmed_class} <<< (Confidence: {confidence*100:.1f}%)")
                        print(f" [DISPATCH] Opening {target_bin} BIN on ESP32-S3...")
                        print("#" * 70)

                        success, status_msg = esp32.send_bin_command(confirmed_class)
                        last_action_msg = f"{target_bin} ({status_msg})"

                        stats["total_dispatched"] += 1
                        stats["counts"][confirmed_class] += 1

                        in_cooldown = True
                        cooldown_end_time = current_time + cooldown_sec
                else:
                    stability_streak = 0
                    current_candidate_class = None

            # Render HUD
            hud_state = {
                "pred_class": pred_class,
                "confidence": confidence,
                "probs": probs,
                "conf_thresh": conf_thresh,
                "stability_streak": stability_streak,
                "required_frames": required_frames,
                "in_cooldown": in_cooldown,
                "cooldown_end": cooldown_end_time,
                "demo_mode": esp32.demo_mode,
                "esp32_connected": esp32.is_connected,
                "serial_port": esp32.port,
                "last_action_msg": last_action_msg,
                "fps": fps,
                "cam_idx": cam_idx,
                "cam_backend": cam_backend
            }
            HUDOverlay.draw_hud(frame, hud_state)

            cv2.imshow("Smart Garbage Classification System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                break
            elif key == ord('r') or key == ord('R'):
                stability_streak = 0
                current_candidate_class = None
                print("[USER] Stability streak reset.")
            elif key == ord('d') or key == ord('D'):
                esp32.demo_mode = not esp32.demo_mode
                mode_str = "ENABLED (Simulation)" if esp32.demo_mode else "DISABLED (Real Serial)"
                print(f"[USER] Demo Mode: {mode_str}")
                if not esp32.demo_mode and not esp32.is_connected:
                    esp32.connect()
            elif key == ord('c') or key == ord('C'):
                print("[USER] Reconnecting camera...")
                cap.release()
                cap, cam_idx, cam_backend = initialize_camera()

    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        esp32.close()

        os.makedirs("results", exist_ok=True)
        stats["session_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open("results/session_stats.json", "w") as f:
            json.dump(stats, f, indent=4)

        print("\n[SESSION COMPLETE] Session log written to results/session_stats.json\n")


def main():
    parser = argparse.ArgumentParser(description="Smart Garbage Classification Live Camera App")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to TFLite or Keras model")
    parser.add_argument("--camera", type=int, default=None, help="OpenCV webcam index override (e.g. 0, 1, 2)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD, help="Confidence threshold (0.0 to 1.0)")
    parser.add_argument("--streak", type=int, default=DEFAULT_CONSECUTIVE_FRAMES, help="Consecutive frames for confirmation")
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COMMAND_COOLDOWN, help="Cooldown in seconds after opening bin")
    parser.add_argument("--port", type=str, default=DEFAULT_SERIAL_PORT, help="ESP32 COM Port (e.g., COM3)")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD_RATE, help="ESP32 Baud Rate (default: 115200)")
    parser.add_argument("--demo", action="store_true", help="Launch directly in Demo Hardware Mode")
    args = parser.parse_args()

    run_garbage_classifier(
        model_path=args.model,
        preferred_camera=args.camera,
        conf_thresh=args.conf,
        required_frames=args.streak,
        cooldown_sec=args.cooldown,
        serial_port=args.port,
        baud_rate=args.baud,
        demo_mode=args.demo
    )


if __name__ == "__main__":
    main()
