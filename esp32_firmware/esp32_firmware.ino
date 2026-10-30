/*
 * ==============================================================================
 * SMART GARBAGE CLASSIFICATION SYSTEM - ESP32-S3 HARDWARE CONTROLLER FIRMWARE
 * ==============================================================================
 * Target Board: Seeed Studio XIAO ESP32-S3 / ESP32-S3 Dev Module
 * Communication: Serial UART (115200 baud)
 * Commands Received from Python Software:
 *   "ORGANIC\n"        -> Opens Organic Bin Servo
 *   "RECYCLABLE\n"     -> Opens Recyclable Bin Servo
 *   "NON_RECYCLABLE\n" -> Opens Non-Recyclable Bin Servo
 *   "HAZARDOUS\n"      -> Opens Hazardous Bin Servo
 * ==============================================================================
 */

#include <ESP32Servo.h>
#include <WiFi.h>
#include <WebServer.h>

// Set these for the Wi-Fi network used by the ESP32. The IP is shown in Serial Monitor.
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
WebServer server(80);

// Servo Pin Assignments (Configurable according to hardware PCB layout)
#define SERVO_PIN_ORGANIC        4
#define SERVO_PIN_RECYCLABLE     5
#define SERVO_PIN_NON_RECYCLABLE 6
#define SERVO_PIN_HAZARDOUS      7

// Auxiliary Relays & Actuators
#define RELAY_UV_C_PIN           8
#define RELAY_HEATER_PIN         9
#define RELAY_VACUUM_PIN         10
#define STATUS_LED_PIN           21

// Servo Position Constants
#define SERVO_CLOSED_ANGLE       0
#define SERVO_OPEN_ANGLE         90
#define BIN_OPEN_HOLD_MS         15000 // Existing 15-second automatic return

Servo servoOrganic;
Servo servoRecyclable;
Servo servoNonRecyclable;
Servo servoHazardous;

void handleBinCycle(Servo &targetServo, const char* binName);

void addCorsHeaders() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.sendHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
  server.sendHeader("Access-Control-Allow-Headers", "Content-Type");
}

void handleOptions() {
  addCorsHeaders();
  server.send(204);
}

void handleStatus() {
  addCorsHeaders();
  String body = "{\"connected\":true,\"ip\":\"" + WiFi.localIP().toString() +
                "\",\"endpoints\":[\"/api/select\",\"/api/toggle\"]}";
  server.send(200, "application/json", body);
}

void handleSelect() {
  const String type = server.arg("type");
  Servo* target = nullptr;
  const char* name = nullptr;

  // The HTTP API exposes the three physical bins. Hazardous is intentionally
  // routed to the existing inorganic bin rather than claiming a fourth bin.
  if (type.equalsIgnoreCase("organic")) {
    target = &servoOrganic;
    name = "ORGANIC BIN";
  } else if (type.equalsIgnoreCase("reusable")) {
    target = &servoRecyclable;
    name = "REUSABLE BIN";
  } else if (type.equalsIgnoreCase("inorganic")) {
    target = &servoNonRecyclable;
    name = "INORGANIC BIN";
  }

  addCorsHeaders();
  if (!target) {
    server.send(400, "application/json", "{\"success\":false,\"error\":\"type must be organic, reusable, or inorganic\"}");
    return;
  }

  handleBinCycle(*target, name);
  server.send(200, "application/json", "{\"success\":true,\"acknowledged\":true}");
}

void configureWebApi() {
  server.on("/api/status", HTTP_GET, handleStatus);
  server.on("/api/select", HTTP_GET, handleSelect);
  server.on("/api/toggle", HTTP_GET, handleSelect);
  server.onNotFound([]() {
    if (server.method() == HTTP_OPTIONS) {
      handleOptions();
    } else {
      addCorsHeaders();
      server.send(404, "application/json", "{\"success\":false,\"error\":\"Not found\"}");
    }
  });
  server.begin();
}

void setup() {
  // Initialize Serial interface
  Serial.begin(115200);
  delay(500);

  // Allocate hardware PWM timers for servos
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  // Set standard 50Hz PWM frequency for SG90 / MG996R Servos
  servoOrganic.setPeriodHertz(50);
  servoRecyclable.setPeriodHertz(50);
  servoNonRecyclable.setPeriodHertz(50);
  servoHazardous.setPeriodHertz(50);

  // Attach servos to respective GPIO pins
  servoOrganic.attach(SERVO_PIN_ORGANIC, 500, 2400);
  servoRecyclable.attach(SERVO_PIN_RECYCLABLE, 500, 2400);
  servoNonRecyclable.attach(SERVO_PIN_NON_RECYCLABLE, 500, 2400);
  servoHazardous.attach(SERVO_PIN_HAZARDOUS, 500, 2400);

  // Configure Relays & Status LED
  pinMode(STATUS_LED_PIN, OUTPUT);
  pinMode(RELAY_UV_C_PIN, OUTPUT);
  pinMode(RELAY_HEATER_PIN, OUTPUT);
  pinMode(RELAY_VACUUM_PIN, OUTPUT);

  digitalWrite(STATUS_LED_PIN, HIGH);
  digitalWrite(RELAY_UV_C_PIN, LOW);
  digitalWrite(RELAY_HEATER_PIN, LOW);
  digitalWrite(RELAY_VACUUM_PIN, LOW);

  // Set initial position: All bins closed
  resetAllServos();

  Serial.println("[ESP32-S3 READY] Smart Garbage Classification Firmware Initialized.");
  Serial.println("[AWAITING COMMANDS]: ORGANIC | RECYCLABLE | NON_RECYCLABLE | HAZARDOUS");

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("[WIFI] Connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("[WIFI] ESP32 IP address: ");
  Serial.println(WiFi.localIP());
  configureWebApi();
}

void loop() {
  server.handleClient();
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim(); // Remove whitespace / carriage returns

    if (command.length() == 0) return;

    Serial.print("[ESP32 ACK] Command received: ");
    Serial.println(command);

    if (command.equalsIgnoreCase("ORGANIC")) {
      handleBinCycle(servoOrganic, "ORGANIC BIN");
    } 
    else if (command.equalsIgnoreCase("RECYCLABLE")) {
      handleBinCycle(servoRecyclable, "RECYCLABLE BIN");
    } 
    else if (command.equalsIgnoreCase("NON_RECYCLABLE") || command.equalsIgnoreCase("NON-RECYCLABLE")) {
      handleBinCycle(servoNonRecyclable, "NON-RECYCLABLE BIN");
    } 
    else if (command.equalsIgnoreCase("HAZARDOUS")) {
      handleBinCycle(servoHazardous, "HAZARDOUS BIN");
    } 
    else {
      Serial.print("[ESP32 ERROR] Unknown command: ");
      Serial.println(command);
    }
  }
}

void handleBinCycle(Servo &targetServo, const char* binName) {
  Serial.print("[ACTUATOR] Opening ");
  Serial.println(binName);
  
  digitalWrite(STATUS_LED_PIN, LOW); // Flash LED
  targetServo.write(SERVO_OPEN_ANGLE);
  delay(BIN_OPEN_HOLD_MS);

  Serial.print("[ACTUATOR] Closing ");
  Serial.println(binName);
  targetServo.write(SERVO_CLOSED_ANGLE);
  digitalWrite(STATUS_LED_PIN, HIGH);
  
  Serial.println("[ACTUATOR COMPLETE] Ready for next command.");
}

void resetAllServos() {
  servoOrganic.write(SERVO_CLOSED_ANGLE);
  servoRecyclable.write(SERVO_CLOSED_ANGLE);
  servoNonRecyclable.write(SERVO_CLOSED_ANGLE);
  servoHazardous.write(SERVO_CLOSED_ANGLE);
}
