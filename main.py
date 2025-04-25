import numpy as np
import matplotlib.pyplot as plt

# Simulated human body (plant) model
def plant_model(current_saturation, valve_opening, natural_oxygen_loss=0.01):
    # Simplified model: saturation increases with valve opening but decreases naturally
    max_saturation = 100  # maximum oxygen saturation level
    saturation_change = 0.5 * valve_opening - natural_oxygen_loss
    new_saturation = current_saturation + saturation_change
    return max(0, min(max_saturation, new_saturation))

# PID controller
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0

    def control(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return max(0, min(1, output))  # Clamp valve opening between 0 and 1

# Simulation parameters
setpoint = 95  # desired oxygen saturation level (%)
initial_saturation = 90  # initial oxygen saturation level (%)
simulation_time = 100  # total simulation time (seconds)
time_step = 0.1  # time step (seconds)

# PID controller gains
kp = 1.0
ki = 0.1
kd = 0.05

# Initialize
pid = PIDController(kp, ki, kd)
current_saturation = initial_saturation
valve_opening = 0
time = np.arange(0, simulation_time, time_step)
saturation_values = []
valve_opening_values = []

# Simulation loop
for t in time:
    error = setpoint - current_saturation
    valve_opening = pid.control(error, time_step)
    current_saturation = plant_model(current_saturation, valve_opening)
    saturation_values.append(current_saturation)
    valve_opening_values.append(valve_opening)

# Plot results
plt.figure(figsize=(10, 6))

# Plot oxygen saturation
plt.subplot(2, 1, 1)
plt.plot(time, saturation_values, label="Oxygen Saturation (%)", color="blue")
plt.axhline(setpoint, color="red", linestyle="--", label="Setpoint")
plt.xlabel("Time (s)")
plt.ylabel("Oxygen Saturation (%)")
plt.title("Oxygen Saturation Control")
plt.legend()

# Plot valve opening
plt.subplot(2, 1, 2)
plt.plot(time, valve_opening_values, label="Valve Opening (%)", color="green")
plt.xlabel("Time (s)")
plt.ylabel("Valve Opening (0-1)")
plt.title("Valve Opening")
plt.legend()

plt.tight_layout()
plt.show()
