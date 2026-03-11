#include <Servo.h>

// UART settings
const unsigned long BAUD = 115200;

// Servo setup (edit pin as needed)
const int SERVO_PIN = 9;
Servo gateServo;

// Example positions
const int SERVO_HOME = 90;
const int SERVO_ACTIVE = 140;

void sendStart() {
  Serial.println("START");
}

void sendServoMove() {
  Serial.println("SERVO");
}

void setup() {
  Serial.begin(BAUD);
  gateServo.attach(SERVO_PIN);
  gateServo.write(SERVO_HOME);
  delay(500);
  sendStart();
}

void loop() {
  // Example: move servo every 10 seconds and notify RPi
  delay(10000);
  gateServo.write(SERVO_ACTIVE);
  delay(400);
  sendServoMove();
  delay(600);
  gateServo.write(SERVO_HOME);
}
