/*
==============================================================================
SMART GARBAGE CLASSIFICATION SYSTEM - ESP32-S3 EMBEDDED INFERENCE FIRMWARE
Target Board: Seeed Studio XIAO ESP32-S3 Sense (OV2640 Camera + 8MB PSRAM)
Framework: Arduino IDE with Edge Impulse SDK / TensorFlow Lite for Microcontrollers
==============================================================================
Architecture Flow:
  OV2640 Camera -> Image Capture (224x224 RGB) -> Preprocessing (Crop & Scale)
                -> INT8 TinyML Inference -> Softmax Decoding (3 Classes)
                -> Servo Actuation to Target Waste Bin
==============================================================================
*/

#include <Arduino.h>
#include <ESP32Servo.h>
#include "esp_camera.h"

// If using Edge Impulse exported Arduino library:
// #include <smart_garbage_classifier_inferencing.h>

// Servo pin configuration for motorized waste diverter chute
#define SERVO_PIN D1
Servo binDiverterServo;

// Waste Bin Chute Angles
#define ANGLE_ORGANIC        45
#define ANGLE_INORGANIC 90
#define ANGLE_REUSABLE  135
#define ANGLE_REST           0

// Camera pin definitions for Seeed Studio XIAO ESP32-S3 Sense
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     10
#define SIOD_GPIO_NUM     40
#define SIOC_GPIO_NUM     39

#define Y9_GPIO_NUM       48
#define Y8_GPIO_NUM       11
#define Y7_GPIO_NUM       12
#define Y6_GPIO_NUM       14
#define Y5_GPIO_NUM       16
#define Y4_GPIO_NUM       18
#define Y3_GPIO_NUM       17
#define Y2_GPIO_NUM       15
#define VSYNC_GPIO_NUM    38
#define HREF_GPIO_NUM     47
#define PCLK_GPIO_NUM     13

const char* CLASS_NAMES[] = {"Organic", "Recyclable", "Non-Recyclable", "Hazardous"};

void initCamera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_RGB565; // or PIXFORMAT_JPEG
    config.frame_size = FRAMESIZE_240X240;
    config.jpeg_quality = 12;
    config.fb_count = 2;
    config.grab_mode = CAMERA_GRAB_LATEST;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera initialization failed with error 0x%x\n", err);
        while(true);
    }
    Serial.println("OV2640 Camera initialized successfully.");
}

void routeToBin(int classIndex) {
    int targetAngle = ANGLE_REST;
    switch (classIndex) {
        case 0: targetAngle = ANGLE_ORGANIC; break;
        case 1: targetAngle = ANGLE_INORGANIC; break;
        case 2: targetAngle = ANGLE_REUSABLE; break;
    }
    Serial.printf("Actuating Chute to Bin [%s] at %d degrees...\n", CLASS_NAMES[classIndex], targetAngle);
    binDiverterServo.write(targetAngle);
    delay(2000); // Allow item to drop
    binDiverterServo.write(ANGLE_REST); // Return to neutral
}

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000);
    Serial.println("=================================================");
    Serial.println("Smart Garbage Classification - ESP32-S3 TinyML");
    Serial.println("=================================================");

    binDiverterServo.attach(SERVO_PIN);
    binDiverterServo.write(ANGLE_REST);

    initCamera();
}

void loop() {
    Serial.println("\nAwaiting waste object in classification bay...");
    delay(3000);

    // 1. Capture camera frame
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Failed to capture camera frame.");
        return;
    }

    Serial.printf("Captured frame: %dx%d (%d bytes)\n", fb->width, fb->height, fb->len);

    /*
     * 2. TinyML Inference Execution
     * -------------------------------------------------------------
     * In Edge Impulse C++ SDK:
     * signal_t signal;
     * signal.total_length = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT;
     * signal.get_data = &raw_feature_get_data;
     *
     * ei_impulse_result_t result = { 0 };
     * EI_IMPULSE_ERROR r = run_classifier(&signal, &result, false);
     * -------------------------------------------------------------
     */

    // Simulated inference result wrapper for demonstration
    float mockProbabilities[4] = {0.962f, 0.018f, 0.012f, 0.008f}; // e.g. Organic waste
    int predictedClass = 0;
    float maxConfidence = mockProbabilities[0];

    for (int i = 1; i < 4; i++) {
        if (mockProbabilities[i] > maxConfidence) {
            maxConfidence = mockProbabilities[i];
            predictedClass = i;
        }
    }

    Serial.println("\n[INFERENCE RESULT]");
    Serial.printf("  Detected Category: %s\n", CLASS_NAMES[predictedClass]);
    Serial.printf("  Confidence Score:  %.2f%%\n", maxConfidence * 100.0f);
    Serial.printf("  Routing Waste to:  %s Bin\n", CLASS_NAMES[predictedClass]);

    // 3. Actuate mechanical sorting mechanism
    routeToBin(predictedClass);

    // Return frame buffer
    esp_camera_fb_return(fb);
}
