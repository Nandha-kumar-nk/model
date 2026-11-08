// ==============================================================================
// SMART GARBAGE CLASSIFICATION SYSTEM - APPLICATION CONTROLLER (app.js)
// In-Browser WebRTC Live Webcam Capture, AI Pipeline & Telemetry Controller
// ==============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // -------------------------------------------------------------
    // 1. DOM Elements
    // -------------------------------------------------------------
    const video = document.getElementById('camera');
    const captureCanvas = document.getElementById('captureCanvas');
    const videoViewport = document.getElementById('videoViewport');
    const videoPlaceholder = document.getElementById('videoPlaceholder');
    const scanLaser = document.getElementById('scanLaser');
    const reticleBox = document.getElementById('reticleBox');

    // Camera Status DOM
    const statusDot = document.getElementById('statusDot');
    const cameraStatusText = document.getElementById('cameraStatusText');
    const cameraAlertBox = document.getElementById('cameraAlertBox');
    const alertTitle = document.getElementById('alertTitle');
    const alertMessage = document.getElementById('alertMessage');
    const btnDismissAlert = document.getElementById('btnDismissAlert');

    // Control Buttons
    const btnStartCamera = document.getElementById('btnStartCamera');
    const btnStopCamera = document.getElementById('btnStopCamera');
    const btnCaptureFrame = document.getElementById('btnCaptureFrame');
    const btnToggleRealtime = document.getElementById('btnToggleRealtime');
    const toggleRealtimeText = document.getElementById('toggleRealtimeText');

    // File Upload Alternative
    const btnToggleUploadTab = document.getElementById('btnToggleUploadTab');
    const uploadDropzone = document.getElementById('uploadDropzone');
    const fileInput = document.getElementById('fileInput');

    // Result Prediction Elements
    const classificationStatusBadge = document.getElementById('classificationStatusBadge');
    const predCategoryName = document.getElementById('predCategoryName');
    const predConfidenceBadge = document.getElementById('predConfidenceBadge');
    const predItemDetail = document.getElementById('predItemDetail');
    const predictionHighlightBox = document.getElementById('predictionHighlightBox');

    // Stability Meter Elements
    const stabilityText = document.getElementById('stabilityText');
    const pips = [
        document.getElementById('pip1'),
        document.getElementById('pip2'),
        document.getElementById('pip3'),
        document.getElementById('pip4'),
        document.getElementById('pip5')
    ];

    // Probability Bars & Values
    const probBars = {
        ORGANIC: { bar: document.getElementById('barOrganic'), val: document.getElementById('valOrganic') },
        INORGANIC: { bar: document.getElementById('barInorganic'), val: document.getElementById('valInorganic') },
        REUSABLE: { bar: document.getElementById('barReusable'), val: document.getElementById('valReusable') }
    };

    // Recommended Bin & Hardware Command
    const binTargetName = document.getElementById('binTargetName');
    const binTargetDesc = document.getElementById('binTargetDesc');
    const binRecommendationBox = document.getElementById('binRecommendationBox');
    const hardwareCommandDisplay = document.getElementById('hardwareCommandDisplay');
    const hwStatusTag = document.getElementById('hwStatusTag');
    const esp32IpAddress = document.getElementById('esp32IpAddress');
    const btnConnectEsp32 = document.getElementById('btnConnectEsp32');
    const btnTestEsp32 = document.getElementById('btnTestEsp32');
    const btnDisconnectEsp32 = document.getElementById('btnDisconnectEsp32');
    const esp32ConnectionStatus = document.getElementById('esp32ConnectionStatus');
    const esp32ConnectionMessage = document.getElementById('esp32ConnectionMessage');
    const hardwareLastPrediction = document.getElementById('hardwareLastPrediction');
    const hardwareLastConfidence = document.getElementById('hardwareLastConfidence');
    const hardwareLastCommand = document.getElementById('hardwareLastCommand');
    const hardwareSelectedBin = document.getElementById('hardwareSelectedBin');
    const hardwareCommandResult = document.getElementById('hardwareCommandResult');
    const aiModelStatus = document.getElementById('aiModelStatus');

    // Mobile nav toggle
    const mobileToggle = document.getElementById('mobileToggle');
    const navLinks = document.getElementById('navLinks');

    // -------------------------------------------------------------
    // 2. Application State Variables
    // -------------------------------------------------------------
    let mediaStream = null;
    let isRealtimeRunning = false;
    let realtimeTimer = null;
    let isClassifying = false;

    // Consecutive Frame Stability Tracker
    const REQUIRED_CONSECUTIVE_FRAMES = 5;
    const COMMAND_COOLDOWN_MS = 5000;
    let currentCandidateClass = null;
    let stabilityStreak = 0;
    let lastDispatchedClass = null;
    let lastCommandAt = 0;
    let esp32BaseUrl = null;

    function resetStabilityState() {
        currentCandidateClass = null;
        stabilityStreak = 0;
        lastDispatchedClass = null;
        stabilityText.textContent = `0 / ${REQUIRED_CONSECUTIVE_FRAMES} Frames`;
        pips.forEach(pip => {
            pip.classList.remove('active');
            pip.style.backgroundColor = '';
        });
    }

    // Mobile Navigation Toggle
    if (mobileToggle && navLinks) {
        mobileToggle.addEventListener('click', () => {
            navLinks.classList.toggle('mobile-open');
        });
    }

    // Dismiss Alert Banner
    if (btnDismissAlert && cameraAlertBox) {
        btnDismissAlert.addEventListener('click', () => {
            cameraAlertBox.classList.add('hidden');
        });
    }

    function setHardwareStatus(state, message) {
        const labels = {
            connected: "Connected", connecting: "Connecting", disconnected: "Disconnected", error: "Error"
        };
        esp32ConnectionStatus.dataset.state = state;
        esp32ConnectionStatus.innerHTML = `<span class="status-dot ${state === 'connected' ? 'connected' : state === 'error' ? 'error' : state === 'connecting' ? 'connecting' : 'off'}"></span> ${labels[state]}`;
        esp32ConnectionMessage.textContent = message;
    }

    function validIpAddress(value) {
        const parts = value.trim().split(".");
        return parts.length === 4 && parts.every(part => /^\d{1,3}$/.test(part) && Number(part) >= 0 && Number(part) <= 255);
    }

    async function requestEsp32(path) {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 20000);
        try {
            const response = await fetch(`${esp32BaseUrl}${path}`, {
                method: "GET",
                headers: { Accept: "application/json" },
                signal: controller.signal
            });
            if (!response.ok) throw new Error(`ESP32 returned HTTP ${response.status}`);
            return await response.json();
        } finally {
            clearTimeout(timeout);
        }
    }

    async function testEsp32Connection() {
        const address = esp32IpAddress.value.trim();
        if (!validIpAddress(address)) {
            esp32BaseUrl = null;
            btnTestEsp32.disabled = true;
            btnDisconnectEsp32.disabled = true;
            setHardwareStatus("error", "Enter a valid IPv4 address, for example 192.168.1.100.");
            return false;
        }
        esp32BaseUrl = `http://${address}`;
        setHardwareStatus("connecting", "Checking /api/status...");
        try {
            const status = await requestEsp32("/api/status");
            setHardwareStatus("connected", `Connected to ${address}. /api/status responded.`);
            btnTestEsp32.disabled = false;
            btnDisconnectEsp32.disabled = false;
            return Boolean(status);
        } catch (error) {
            esp32BaseUrl = null;
            btnTestEsp32.disabled = true;
            btnDisconnectEsp32.disabled = true;
            setHardwareStatus("error", `Connection failed: ${error.name === "AbortError" ? "request timed out" : error.message}`);
            return false;
        }
    }

    function disconnectEsp32() {
        esp32BaseUrl = null;
        btnTestEsp32.disabled = true;
        btnDisconnectEsp32.disabled = true;
        setHardwareStatus("disconnected", "Disconnected. Camera and AI classification can run independently.");
    }

    async function sendHardwareCommand(categoryKey, confidence) {
        const category = WASTE_CATEGORIES[categoryKey];
        if (!esp32BaseUrl) {
            hardwareCommandResult.textContent = "Not sent: ESP32 disconnected";
            hwStatusTag.textContent = "DISCONNECTED";
            return;
        }
        if (Date.now() - lastCommandAt < COMMAND_COOLDOWN_MS) {
            hardwareCommandResult.textContent = "Held: command cooldown";
            hwStatusTag.textContent = "COOLDOWN";
            return;
        }

        hardwareLastCommand.textContent = `/api/select?type=${category.physicalBin}`;
        hardwareSelectedBin.textContent = category.physicalBin;
        hwStatusTag.textContent = "AWAITING API";
        try {
            const result = await requestEsp32(`/api/select?type=${encodeURIComponent(category.physicalBin)}`);
            lastCommandAt = Date.now();
            hardwareCommandResult.textContent = result && result.success === false ? "API rejected command" : "API acknowledged";
            hwStatusTag.textContent = result && result.success === false ? "API ERROR" : "ACKNOWLEDGED";
            hwStatusTag.className = result && result.success === false ? "hw-status-tag" : "hw-status-tag active";
        } catch (error) {
            hardwareCommandResult.textContent = `Error: ${error.message}`;
            hwStatusTag.textContent = "NETWORK ERROR";
            hwStatusTag.className = "hw-status-tag";
            esp32BaseUrl = null;
            btnTestEsp32.disabled = true;
            btnDisconnectEsp32.disabled = true;
            setHardwareStatus("error", `Hardware request failed: ${error.message}`);
        }
    }

    // -------------------------------------------------------------
    // 3. WebRTC Camera Initialization & Management
    // -------------------------------------------------------------
    function setCameraStatus(statusKey, customError = null) {
        statusDot.className = 'status-dot';

        switch (statusKey) {
            case 'CONNECTED':
                statusDot.classList.add('connected');
                cameraStatusText.textContent = 'Camera: CONNECTED';
                cameraStatusText.style.color = 'var(--organic-color)';
                videoPlaceholder.style.display = 'none';
                btnStartCamera.disabled = true;
                btnStopCamera.disabled = false;
                btnCaptureFrame.disabled = false;
                btnToggleRealtime.disabled = false;
                if (cameraAlertBox) cameraAlertBox.classList.add('hidden');
                break;

            case 'OFF':
                statusDot.classList.add('off');
                cameraStatusText.textContent = 'Camera: OFF';
                cameraStatusText.style.color = 'var(--text-secondary)';
                videoPlaceholder.style.display = 'flex';
                btnStartCamera.disabled = false;
                btnStopCamera.disabled = true;
                btnCaptureFrame.disabled = true;
                btnToggleRealtime.disabled = true;
                stopRealtimeClassification();
                break;

            case 'PERMISSION_DENIED':
                statusDot.classList.add('error');
                cameraStatusText.textContent = 'Camera: PERMISSION DENIED';
                cameraStatusText.style.color = 'var(--hazardous-color)';
                showCameraAlert(
                    "Camera Permission Denied",
                    customError || "Chrome blocked camera access. Open the lock icon in the address bar, set Camera to Allow, check Windows Settings > Privacy & security > Camera, then reload this page."
                );
                break;

            case 'NOT_FOUND':
                statusDot.classList.add('error');
                cameraStatusText.textContent = 'Camera: NOT FOUND';
                cameraStatusText.style.color = 'var(--hazardous-color)';
                showCameraAlert(
                    "No Camera Detected",
                    customError || "No camera was found. Check that the Windows built-in webcam is enabled and its privacy shutter is open. Then enable Windows Settings > Privacy & security > Camera > Camera access and allow Chrome to use it."
                );
                break;

            case 'ERROR':
                statusDot.classList.add('error');
                cameraStatusText.textContent = 'Camera: ERROR';
                cameraStatusText.style.color = 'var(--hazardous-color)';
                showCameraAlert("Camera Error", customError || "An unexpected error occurred while accessing the webcam.");
                break;
        }
    }

    function showCameraAlert(title, message) {
        if (!cameraAlertBox) return;
        alertTitle.textContent = title;
        alertMessage.textContent = message;
        cameraAlertBox.classList.remove('hidden');
    }

    /**
     * Activates the browser's default available webcam.
     * No camera ID or facing-mode constraint is selected by the application.
     */
    async function startCamera() {
        console.log("[WEBRTC] Requesting camera access...");

        // Stop an earlier session before requesting another one.
        stopCamera();
        setCameraStatus('OFF');

        const isLocalhost = ['localhost', '127.0.0.1', '::1'].includes(window.location.hostname);
        if (!window.isSecureContext && !isLocalhost) {
            console.error("[WEBRTC ERROR] Camera requires localhost or HTTPS.");
            setCameraStatus('ERROR', "Camera access requires a secure context. Run this site from http://localhost or HTTPS.");
            return;
        }

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia ||
            !navigator.mediaDevices.enumerateDevices) {
            console.error("[WEBRTC ERROR] getUserMedia not supported in this browser context.");
            setCameraStatus('ERROR', "Camera access is unavailable in this browser. Run the site from http://localhost or HTTPS and use a current browser.");
            return;
        }

        try {
            const devicesBeforePermission = await navigator.mediaDevices.enumerateDevices();
            const camerasBeforePermission = devicesBeforePermission.filter(
                device => device.kind === 'videoinput'
            );
            console.log(`[WEBRTC] Available video inputs before permission: ${camerasBeforePermission.length}`);
        } catch (enumerateError) {
            // Device enumeration may be restricted before permission is granted.
            console.warn("[WEBRTC] Could not enumerate devices before permission:", enumerateError);
        }

        try {
            // Let the browser select its default available camera.
            mediaStream = await navigator.mediaDevices.getUserMedia({
                video: true,
                audio: false
            });
        } catch (cameraError) {
            console.error("[WEBRTC ERROR] Camera access denied or failed:", cameraError);

            if (cameraError.name === 'NotAllowedError' || cameraError.name === 'PermissionDeniedError') {
                setCameraStatus(
                    'PERMISSION_DENIED',
                    "Chrome or Windows denied camera access. In Chrome, open the lock icon > Camera > Allow. Then enable Windows Settings > Privacy & security > Camera > Camera access and allow Chrome."
                );
            } else if (cameraError.name === 'NotFoundError' || cameraError.name === 'DevicesNotFoundError') {
                let cameraCount = 0;
                try {
                    const devicesAfterFailure = await navigator.mediaDevices.enumerateDevices();
                    cameraCount = devicesAfterFailure.filter(device => device.kind === 'videoinput').length;
                } catch (enumerateError) {
                    console.warn("[WEBRTC] Could not inspect devices after camera failure:", enumerateError);
                }
                setCameraStatus(
                    'NOT_FOUND',
                    cameraCount === 0
                        ? "No video input was detected by the browser. Check that the Windows built-in webcam is enabled, its privacy shutter is open, and Windows Settings > Privacy & security > Camera allows camera access for Chrome."
                        : "Chrome can see a camera, but could not open it. Check Chrome's site camera permission, close other apps using the webcam, and reload the page."
                );
            } else if (cameraError.name === 'NotReadableError' || cameraError.name === 'TrackStartError') {
                setCameraStatus(
                    'ERROR',
                    "The camera was found but could not be started. Close Teams, Zoom, Camera, or other apps using it, check the Windows webcam driver, then reload this page."
                );
            } else {
                setCameraStatus('ERROR', `${cameraError.name}: ${cameraError.message}`);
            }
            return;
        }

        try {
            const devicesAfterPermission = await navigator.mediaDevices.enumerateDevices();
            const camerasAfterPermission = devicesAfterPermission.filter(
                device => device.kind === 'videoinput'
            );
            console.log(`[WEBRTC] Available video inputs after permission: ${camerasAfterPermission.length}`);
        } catch (enumerateError) {
            console.warn("[WEBRTC] Could not enumerate devices after permission:", enumerateError);
        }

        video.onloadedmetadata = async () => {
            try {
                await video.play();
                setCameraStatus('CONNECTED');
                console.log(`[WEBRTC] Camera live: ${video.videoWidth}x${video.videoHeight}px`);
            } catch (playError) {
                console.error("[WEBRTC ERROR] Video playback could not start:", playError);
                if (mediaStream) {
                    mediaStream.getTracks().forEach(track => track.stop());
                    mediaStream = null;
                }
                video.srcObject = null;
                setCameraStatus('ERROR', `Camera stream opened but video playback failed: ${playError.message}`);
            }
        };
        // Attach the real stream to the existing video element for AI frame capture.
        video.srcObject = mediaStream;
    }

    /**
     * Safely stops every track of the media stream
     */
    function stopCamera() {
        if (mediaStream) {
            console.log("[WEBRTC] Stopping media stream tracks...");
            mediaStream.getTracks().forEach(track => {
                track.stop();
            });
            mediaStream = null;
        }

        video.srcObject = null;
        resetStabilityState();
        setCameraStatus('OFF');
    }

    // -------------------------------------------------------------
    // 4. Frame Capture & AI Inference Pipeline
    // -------------------------------------------------------------
    async function captureAndClassifyFrame() {
        if (!mediaStream || video.readyState < 2 || isClassifying) {
            return;
        }

        isClassifying = true;
        try {
            // Draw current video frame to hidden canvas
            const w = video.videoWidth || 640;
            const h = video.videoHeight || 480;
            captureCanvas.width = w;
            captureCanvas.height = h;

            const ctx = captureCanvas.getContext('2d');
            if (!ctx) throw new Error("The browser could not create a capture canvas.");
            ctx.drawImage(video, 0, 0, w, h);

            // Send captured frame to AI Classifier Engine
            const result = await classifyWasteItem(captureCanvas);
            processClassificationResult(result);
        } catch (error) {
            console.error("[INFERENCE ERROR] Frame classification failed:", error);
            processClassificationResult({
                success: false,
                error: error.message || "Could not complete inference."
            });
        } finally {
            isClassifying = false;
        }
    }

    /**
     * Evaluates prediction, updates HUD, tests 80% threshold & manages 5-frame stability
     */
    function processClassificationResult(res) {
        if (!res || !res.success) {
            if (aiModelStatus) aiModelStatus.textContent = "Error";
            classificationStatusBadge.textContent = "MODEL ERROR";
            classificationStatusBadge.className = "badge badge-hazardous";
            predCategoryName.textContent = "Model Offline";
            predItemDetail.textContent = res.error || "Could not complete inference.";
            return;
        }

        const categoryKey = res.predictedClass;
        const confPercent = (res.confidence * 100).toFixed(1);
        const isHighConfidence = res.isHighConfidence; // >= 80%
        hardwareLastPrediction.textContent = res.categoryName || "Unknown";
        hardwareLastConfidence.textContent = `${confPercent}%`;
        if (aiModelStatus) aiModelStatus.textContent = "Loaded";

        // 1. Update Category Header & Confidence
        predCategoryName.textContent = isHighConfidence ? res.categoryName : "Uncertain (Low Confidence)";
        predConfidenceBadge.textContent = `${confPercent}%`;
        predConfidenceBadge.style.backgroundColor = isHighConfidence ? res.themeColor : 'var(--text-muted)';
        
        predItemDetail.textContent = isHighConfidence 
            ? `Detected feature: "${res.detectedFeature || res.categoryName}"`
            : "Please show the waste item clearly in front of the camera.";

        // 2. Update All 4 Probability Progress Bars
        for (const [key, barGroup] of Object.entries(probBars)) {
            const prob = res.probabilities[key] || 0.0;
            const pct = (prob * 100).toFixed(1);
            if (barGroup.bar) barGroup.bar.style.width = `${pct}%`;
            if (barGroup.val) barGroup.val.textContent = `${pct}%`;
        }

        // 3. Five-Consecutive Frame Prediction Stabilization
        if (isHighConfidence) {
            if (categoryKey === currentCandidateClass) {
                stabilityStreak++;
            } else {
                currentCandidateClass = categoryKey;
                stabilityStreak = 1;
            }

            // Update streak pips
            stabilityText.textContent = `${Math.min(stabilityStreak, REQUIRED_CONSECUTIVE_FRAMES)} / ${REQUIRED_CONSECUTIVE_FRAMES} Frames`;
            pips.forEach((pip, idx) => {
                if (idx < stabilityStreak) {
                    pip.classList.add('active');
                    pip.style.backgroundColor = res.themeColor;
                } else {
                    pip.classList.remove('active');
                    pip.style.backgroundColor = '';
                }
            });

            // If 5 consecutive frames confirm the same category
            if (stabilityStreak >= REQUIRED_CONSECUTIVE_FRAMES) {
                classificationStatusBadge.textContent = "CLASSIFICATION CONFIRMED";
                classificationStatusBadge.className = `badge badge-${categoryKey.toLowerCase().replace('_', '-')}`;

                // Update Recommended Action / Bin
                binTargetName.textContent = res.recommendedAction || res.recommendedBin;
                binTargetName.style.color = res.themeColor;
                binTargetDesc.textContent = `Open the ${res.physicalBin} bin lid for physical deposition.`;

                // Update Hardware Command
                const hwCmd = res.hardwareCommand;
                hardwareCommandDisplay.textContent = hwCmd;
                hardwareCommandDisplay.style.color = res.themeColor;
                if (lastDispatchedClass !== categoryKey && Date.now() - lastCommandAt >= COMMAND_COOLDOWN_MS) {
                    lastDispatchedClass = categoryKey;
                    sendHardwareCommand(categoryKey, res.confidence);
                }

                // Record to LocalStorage telemetry for Dashboard
                recordClassificationToStorage(categoryKey, res.confidence);
                resetStabilityState();
            } else {
                classificationStatusBadge.textContent = `ANALYZING (${stabilityStreak}/${REQUIRED_CONSECUTIVE_FRAMES})`;
                classificationStatusBadge.className = "badge";
            }
        } else {
            // Low confidence -> reset streak
            resetStabilityState();

            classificationStatusBadge.textContent = "LOW CONFIDENCE";
            classificationStatusBadge.className = "badge badge-non-recyclable";
            binTargetName.textContent = "STANDBY [LOW CONFIDENCE]";
            binTargetName.style.color = "var(--text-secondary)";
            binTargetDesc.textContent = "Confidence is below 80% threshold. Please reposition the item.";
            hwStatusTag.textContent = "HOLD";
            hwStatusTag.className = "hw-status-tag";
        }
    }

    /**
     * Continuous periodic inference loop (500 ms)
     */
    function toggleRealtimeClassification() {
        if (isRealtimeRunning) {
            stopRealtimeClassification();
        } else {
            startRealtimeClassification();
        }
    }

    function startRealtimeClassification() {
        if (!mediaStream) {
            startCamera().then(() => {
                if (mediaStream) startRealtimeClassification();
            });
            return;
        }

        isRealtimeRunning = true;
        btnToggleRealtime.classList.add('active');
        toggleRealtimeText.textContent = "PAUSE REAL-TIME INFERENCE";
        scanLaser.classList.add('active');

        // Periodically run inference every 500 ms
        realtimeTimer = setInterval(() => {
            captureAndClassifyFrame();
        }, 500);

        console.log("[INFERENCE] Real-time loop active (500ms interval).");
    }

    function stopRealtimeClassification() {
        isRealtimeRunning = false;
        if (realtimeTimer) {
            clearInterval(realtimeTimer);
            realtimeTimer = null;
        }
        btnToggleRealtime.classList.remove('active');
        toggleRealtimeText.textContent = "START REAL-TIME AI INFERENCE";
        scanLaser.classList.remove('active');
    }

    // -------------------------------------------------------------
    // 5. LocalStorage Telemetry Synchronizer for Dashboard
    // -------------------------------------------------------------
    function recordClassificationToStorage(categoryKey, confidence) {
        try {
            const history = JSON.parse(localStorage.getItem('smart_garbage_history') || '[]');
            const counts = JSON.parse(localStorage.getItem('smart_garbage_counts') || JSON.stringify({
                ORGANIC: 0,
                INORGANIC: 0,
                REUSABLE: 0
            }));

            // Increment count
            if (counts[categoryKey] !== undefined) {
                counts[categoryKey] += 1;
            }

            // Append history item
            history.unshift({
                id: Date.now(),
                category: categoryKey,
                categoryName: WASTE_CATEGORIES[categoryKey].name,
                confidence: confidence,
                bin: WASTE_CATEGORIES[categoryKey].bin,
                hardwareCommand: categoryKey,
                timestamp: new Date().toLocaleTimeString()
            });

            // Keep max 50 recent items
            if (history.length > 50) history.pop();

            localStorage.setItem('smart_garbage_history', JSON.stringify(history));
            localStorage.setItem('smart_garbage_counts', JSON.stringify(counts));
            localStorage.setItem('smart_garbage_last_update', Date.now().toString());

            console.log(`[STORAGE] Logged confirmed waste: ${categoryKey}`);
        } catch (e) {
            console.warn("[STORAGE] Failed to write localStorage telemetry:", e);
        }
    }

    // -------------------------------------------------------------
    // 6. Alternative Image Upload Tab
    // -------------------------------------------------------------
    if (btnToggleUploadTab && uploadDropzone) {
        btnToggleUploadTab.addEventListener('click', () => {
            uploadDropzone.classList.toggle('hidden');
        });

        uploadDropzone.addEventListener('click', (event) => {
            if (event.target !== fileInput) fileInput.click();
        });

        const decodeUploadedImage = async (file) => {
            const maxFileSize = 20 * 1024 * 1024;
            const acceptedTypes = new Set(['image/jpeg', 'image/png', 'image/webp']);
            const extension = file.name.toLowerCase().split('.').pop();
            const acceptedExtensions = new Set(['jpg', 'jpeg', 'png', 'webp']);
            const hasAcceptedType = acceptedTypes.has(file.type) || acceptedExtensions.has(extension);
            if (!hasAcceptedType || file.size > maxFileSize) {
                throw new Error('Please select a JPG, JPEG, PNG, or WEBP image smaller than 20 MB.');
            }

            let source;
            let objectUrl;
            try {
                if (typeof createImageBitmap === 'function') {
                    try {
                        source = await createImageBitmap(file, { imageOrientation: 'from-image' });
                    } catch (error) {
                        console.warn('[UPLOAD] ImageBitmap decode with orientation failed; using Image fallback.', error);
                    }
                }
                if (!source) {
                    objectUrl = URL.createObjectURL(file);
                    source = await new Promise((resolve, reject) => {
                        const image = new Image();
                        image.onload = () => resolve(image);
                        image.onerror = () => reject(new Error('The selected image could not be decoded.'));
                        image.src = objectUrl;
                    });
                }

                if (!source.width || !source.height) {
                    throw new Error('The selected image has no usable dimensions.');
                }
                const maxDimension = 2048;
                const scale = Math.min(1, maxDimension / Math.max(source.width, source.height));
                const canvas = document.createElement('canvas');
                canvas.width = Math.max(1, Math.round(source.width * scale));
                canvas.height = Math.max(1, Math.round(source.height * scale));
                const context = canvas.getContext('2d', { alpha: false });
                if (!context) throw new Error('The browser could not create an image canvas.');
                context.fillStyle = '#ffffff';
                context.fillRect(0, 0, canvas.width, canvas.height);
                context.drawImage(source, 0, 0, canvas.width, canvas.height);
                return canvas;
            } finally {
                if (source && typeof source.close === 'function') source.close();
                if (objectUrl) URL.revokeObjectURL(objectUrl);
            }
        };

        fileInput.addEventListener('change', async (e) => {
            const file = e.target.files && e.target.files[0];
            fileInput.value = '';
            if (!file) return;

            try {
                const imageCanvas = await decodeUploadedImage(file);
                const result = await classifyWasteItem(imageCanvas);
                processClassificationResult(result);
            } catch (error) {
                console.error('[UPLOAD ERROR]:', error);
                processClassificationResult({
                    success: false,
                    error: error.message || 'The selected image could not be processed.'
                });
            }
        });
    }

    // -------------------------------------------------------------
    // 7. Event Listeners
    // -------------------------------------------------------------
    btnStartCamera.addEventListener('click', startCamera);
    btnStopCamera.addEventListener('click', stopCamera);
    btnCaptureFrame.addEventListener('click', captureAndClassifyFrame);
    btnToggleRealtime.addEventListener('click', toggleRealtimeClassification);
    btnConnectEsp32.addEventListener('click', testEsp32Connection);
    btnTestEsp32.addEventListener('click', testEsp32Connection);
    btnDisconnectEsp32.addEventListener('click', disconnectEsp32);

    // Initial state
    setCameraStatus('OFF');
});
