int air = 3;
int open = 0;

void setup() {
  pinMode(air, OUTPUT);
  Serial.begin(9600); // Start serial at 9600 baud
  analogWrite(air, open);
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
  analogWrite(air, open);
}
