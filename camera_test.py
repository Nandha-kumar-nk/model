"""
================================================================================
SMART GARBAGE CLASSIFICATION SYSTEM - STANDALONE CAMERA DIAGNOSTIC & TEST TOOL
================================================================================
Purpose:
  - Scans and validates laptop webcam / USB camera connectivity on Windows.
  - Automatically tests Camera Indexes (0, 1, 2).
  - Tries Windows OpenCV capture backends:
      1) cv2.CAP_DSHOW (DirectShow - recommended for Windows webcams)
      2) cv2.CAP_MSMF  (Media Foundation)
      3) Default OpenCV VideoCapture backend
  - Displays live video feed with overlay status.
  - Works 100% independently of the AI model.
================================================================================
"""

import sys
import time
import cv2


def test_and_initialize_camera(max_index_to_test=3):
    """
    Iterates through camera indices and Windows backends to find and open
    the first available webcam.
    
    Returns:
        cap (cv2.VideoCapture or None): The active OpenCV video capture object.
        used_index (int or None): Index of the working camera.
        used_backend (str or None): Backend name used to open the camera.
    """
    print("=" * 75)
    print(" SMART GARBAGE CAMERA DIAGNOSTIC - SCANNING AVAILABLE WEBCAMS")
    print("=" * 75)

    # Windows OpenCV Backends in priority order
    backends_to_try = [
        ("DirectShow (CAP_DSHOW)", cv2.CAP_DSHOW),
        ("Media Foundation (CAP_MSMF)", cv2.CAP_MSMF),
        ("Default OpenCV Backend", None)
    ]

    for index in range(max_index_to_test):
        print(f"\n[SCAN] Checking Camera Index {index}...")

        for backend_name, backend_flag in backends_to_try:
            print(f"  --> Testing Index {index} using {backend_name}...", end=" ", flush=True)
            
            try:
                if backend_flag is not None:
                    cap = cv2.VideoCapture(index, backend_flag)
                else:
                    cap = cv2.VideoCapture(index)

                if cap is not None and cap.isOpened():
                    # Attempt reading a test frame to ensure sensor data is actually streaming
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        print("SUCCESS! (Frame captured)")
                        print("=" * 75)
                        print(f" [CAMERA FOUND] Successfully connected to Camera Index {index} via {backend_name}")
                        print("=" * 75)

                        # Set default 720p resolution
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        return cap, index, backend_name
                    else:
                        print("FAILED (Device opened but failed to read frame)")
                        cap.release()
                else:
                    print("FAILED (Could not open device)")
                    if cap is not None:
                        cap.release()
            except Exception as e:
                print(f"FAILED (Exception: {e})")

    # If all indices and backends fail
    print("\n" + "!" * 75)
    print(" [ERROR] CAMERA NOT FOUND / FAILED TO INITIALIZE")
    print("!" * 75)
    print(" Possible reasons and troubleshooting steps on Windows:")
    print("   1. Camera privacy permission is disabled:")
    print("      -> Open Windows Settings > Privacy & security > Camera")
    print("      -> Turn ON 'Camera access' and 'Let desktop apps access your camera'")
    print("   2. Another application is currently locking the camera:")
    print("      -> Close Microsoft Teams, Zoom, Skype, Windows Camera app, or browser tabs.")
    print("   3. Webcam driver issue or disabled in Device Manager:")
    print("      -> Open Device Manager > Cameras / Imaging devices > Check if webcam is enabled.")
    print("   4. If using an external USB camera, ensure it is firmly plugged in.")
    print("!" * 75 + "\n")

    return None, None, None


def main():
    cap, used_index, used_backend = test_and_initialize_camera(max_index_to_test=3)

    if cap is None:
        print("[TERMINATING] Exiting camera test because no functional webcam was detected.")
        sys.exit(1)

    print("\n[LIVE CAMERA TEST RUNNING]")
    print("  Press 'Q' or 'ESC' on the video window to close the test.\n")

    fps_counter = 0
    fps = 0.0
    fps_timer = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Frame dropped. Retrying...")
                time.sleep(0.03)
                continue

            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                fps = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer = time.time()

            h, w = frame.shape[:2]

            # -------------------------------------------------------------
            # Render Diagnostic Overlay
            # -------------------------------------------------------------
            # Top Banner
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 65), (20, 24, 30), -1)
            cv2.rectangle(overlay, (0, h - 50), (w, h), (20, 24, 30), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

            # Header Text
            cv2.putText(frame, "SMART GARBAGE CAMERA TEST", (20, 36),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Live Hardware Diagnostic Stream", (20, 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 170, 180), 1, cv2.LINE_AA)

            # FPS & Resolution
            res_str = f"Res: {w}x{h} | FPS: {fps:.1f}"
            cv2.putText(frame, res_str, (w - 240, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (41, 211, 255), 1, cv2.LINE_AA)

            # Center Inspection Box
            box_size = 220
            cx, cy = w // 2, h // 2
            cv2.rectangle(frame, (cx - box_size//2, cy - box_size//2),
                          (cx + box_size//2, cy + box_size//2), (46, 204, 113), 2)
            cv2.putText(frame, "CENTER WASTE ITEM HERE", (cx - 110, cy + box_size//2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (46, 204, 113), 1, cv2.LINE_AA)

            # Bottom Status Bar
            cv2.putText(frame, f"Camera Index: {used_index}", (20, h - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "Status: CONNECTED", (170, h - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (46, 204, 113), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Backend: {used_backend}", (20, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 170, 180), 1, cv2.LINE_AA)
            cv2.putText(frame, "Press 'Q' or 'ESC' to Exit Test", (w - 260, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Smart Garbage Classifier - Camera Diagnostic Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:  # 27 is ESC
                print("\n[USER] Exit requested. Camera test completed successfully.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[CAMERA TEST] VideoCapture released. Windows destroyed.\n")


if __name__ == "__main__":
    main()
