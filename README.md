# Smart Garbage Classification System Using Deep Learning

An end-to-end Edge-AI and Computer Vision software system for real-time automated waste classification and smart bin sorting.

This repository provides **two client implementations**:
1. **In-Browser Web Application (Primary)**: HTML5 + CSS3 + Vanilla JavaScript + WebRTC `getUserMedia()` + TensorFlow.js.
2. **Python OpenCV Desktop Engine (Standalone)**: Python 3 + OpenCV + MobileNetV2 / TFLite + PySerial UART hardware link.

---

## 1. Web Application: In-Browser Live Webcam AI

The camera streams **directly inside the web page** using browser WebRTC APIs without any external OpenCV windows.

```
+------------------------------------------------------------------------------------+
|                         IN-BROWSER WEBRTC AI PIPELINE                              |
|                                                                                    |
| [Laptop Webcam]                                                                    |
|        ↓                                                                           |
| [WebRTC navigator.mediaDevices.getUserMedia()]                                     |
|        ↓                                                                           |
| [<video id="camera" autoplay playsinline>]                                         |
|        ↓ (Periodic 500ms Frame Capture)                                            |
| [TensorFlow.js Deep Learning Vision Classifier]                                    |
|        ↓                                                                           |
| [3 Class Probabilities: Organic, Inorganic, Reusable]                              |
|        ↓                                                                           |
| [Confidence Threshold Check (>= 80%)]                                              |
|        ↓                                                                           |
| [5-Consecutive Frame Stability Confirmation Filter]                                |
|        ↓                                                                           |
| [Generated Hardware Command: "organic" / "inorganic" / "reusable"]                |
|        ↓                                                                           |
| [Smart Bins Dashboard & LocalStorage Telemetry Update]                             |
+------------------------------------------------------------------------------------+
```

---

## 2. Three AI Target Classes

| Category | Recommended Bin | Hardware Command | Example Items Included |
| :--- | :--- | :--- | :--- |
| **ORGANIC** | `ORGANIC BIN` | `ORGANIC` | Tomato, potato, banana, banana peel, fruit/veg scraps, food waste |
| **INORGANIC** | `INORGANIC BIN` | `inorganic` | Dirty wrappers, hazardous waste, e-waste, contaminated packaging |
| **REUSABLE** | `REUSABLE BIN` | `reusable` | Plastic bottles, cans, clean cardboard, paper, metal |

---

## 3. How to Run the Web Application (Localhost)

Browsers (Chrome, Edge, Firefox) **require a secure origin (`http://localhost` or `https://`)** to access the laptop camera. Do not double-click `index.html` as a `file://` link.

### Method A: VS Code Live Server (Recommended)
1. Open the project folder in **Visual Studio Code**.
2. Install the **Live Server** extension (`ms-azuretools.vscode-docker` or `ritwickdey.LiveServer`).
3. Right-click [`index.html`](file:///d:/freelance1/smart-garbage-classifier/index.html) &rarr; select **"Open with Live Server"**.
4. Chrome will automatically open `http://localhost:5500/index.html`.
5. Click **START CAMERA**.
6. When prompted, click **Allow** for camera access.

### Method B: Python Built-in Local HTTP Server
```powershell
cd d:\freelance1\smart-garbage-classifier
python -m http.server 8000
```
Open your browser and navigate to: **`http://localhost:8000`**

---

## 4. Live Web Application Controls

- **START CAMERA**: Requests browser camera permissions and streams live video directly inside `<video id="camera">`.
- **STOP CAMERA**: Closes the camera stream and resets the viewport.
- **CAPTURE FRAME**: Captures the current frame to a hidden canvas and runs AI classification immediately.
- **START REAL-TIME AI INFERENCE**: Continuously captures and evaluates frames every **500 ms**.
- **Camera Status Badges**:
  - `Camera: OFF` &mdash; Initial / Standby state.
  - `Camera: CONNECTED` &mdash; Live video stream active.
  - `Camera: PERMISSION DENIED` &mdash; Permission blocked in browser settings.
  - `Camera: NOT FOUND` &mdash; No webcam detected on system.

---

## 5. Browser-to-ESP32 Integration

The classifier page now connects to the ESP32 over Wi-Fi using the IP address entered in
**Hardware Connection — ESP32**. The browser checks `GET /api/status` before showing
Connected and sends stable AI results to:

| AI class | API request | Physical bin |
| :--- | :--- | :--- |
| `ORGANIC` | `/api/select?type=organic` | Organic |
| `INORGANIC` | `/api/select?type=inorganic` | Inorganic |
| `REUSABLE` | `/api/select?type=reusable` | Reusable |

The browser uses an 80% confidence threshold, five consecutive frames, a five-second
cooldown, and a new-detection reset. It does not claim that a servo moved; it reports
only the HTTP/API acknowledgement returned by the ESP32.

### Configure and run the ESP32 API

1. In `esp32_firmware/esp32_firmware.ino`, set `WIFI_SSID` and `WIFI_PASSWORD`.
2. Flash the sketch with the existing servo wiring and libraries (`ESP32Servo`,
   `WiFi`, and `WebServer`).
3. Open Arduino Serial Monitor at 115200 baud and enter the printed `ESP32 IP address`
   into the website.
4. The sketch provides `/api/status`, `/api/select`, and backwards-compatible
   `/api/toggle` routes with CORS headers. The web page uses `/api/select` only.
   The existing servo cycle remains open for 15 seconds before returning home.

Run the site from `http://localhost` (for example with the Python command below or
VS Code Live Server). An HTTPS page cannot make a browser request to an HTTP-only
ESP32 because of mixed-content blocking; use a local HTTP origin, or put an HTTPS
reverse proxy on the same LAN if HTTPS is required.

## 6. Decision Logic & Hardware Commands

1. **Confidence Gating**: Only predictions with confidence $\ge 80\%$ are considered. Below 80%, the system flags `"LOW CONFIDENCE — Please show the waste item clearly."` and holds bin actuation.
2. **5-Frame Stabilization Streak**: Requires the same class to be predicted across **5 consecutive frames** before confirming classification.
3. **Hardware Command Interface**: Sends the mapped `/api/select` request to the ESP32-S3 microcontroller. The trained model is required at `models/model.json`; no MobileNet or keyword fallback is used.

---

## 6. Camera Troubleshooting Checklist

1. **Browser Permissions**: Click the lock 🔒 icon in Chrome's address bar &rarr; *Site settings* &rarr; set *Camera* to **Allow**.
2. **Windows Camera Privacy**: Open *Settings* &rarr; *Privacy & Security* &rarr; *Camera* &rarr; Turn ON **"Let desktop apps access your camera"**.
3. **App Conflicts**: Close Zoom, Microsoft Teams, Google Meet, Skype, or Windows Camera app.
4. **Refresh**: Press `Ctrl + F5` to re-initialize media stream tracks.
5. **Always use Localhost**: Ensure the URL begins with `http://localhost:...` rather than `file://...`.

---

## 7. Python Desktop Engine (Optional Standalone)

If running the standalone Python OpenCV desktop application:
```powershell
# 1. Camera Diagnostic Test
python camera_test.py

# 2. Train Model (Optional)
python train_model.py
python convert_to_tflite.py

# 3. Launch Python OpenCV Live Classifier
python main.py --demo
```
