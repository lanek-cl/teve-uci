int motor1pin1 = 2;
int motor1pin2 = 3;

int open = 0;

void setup() {
  pinMode(motor1pin1, OUTPUT);
  pinMode(motor1pin2, OUTPUT);
  Serial.begin(9600); // Start serial at 9600 baud
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();  // remove whitespace

    int val = input.toInt();  // convert to int
    if (val >= 0 && val <= 255) {
      open = val;
    }
  }

  analogWrite(motor1pin1, open);
  analogWrite(motor1pin2, 0);
}
