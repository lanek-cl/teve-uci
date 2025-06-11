import serial
import time

import serial.tools.list_ports

ports = serial.tools.list_ports.comports()
for port in ports:
    print(port.device)

    
arduino = serial.Serial('COM3', 9600, timeout=1)  # Adjust port!
time.sleep(2)  # Wait for Arduino to reset

def set_open(value):
    if 0 <= value <= 255:
        command = f"{value}\n"
        arduino.write(command.encode('utf-8'))
        print(f"Sent open: {value}")

# Example usage
set_open(120)  # Motor at ~half speed
time.sleep(2)
set_open(255)  # Full speed
time.sleep(2)
set_open(0)    # Stop