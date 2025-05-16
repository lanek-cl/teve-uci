# -*- coding: utf-8 -*-
"""
@file    : main.py
@brief   : Runs main controller simulation
@date    : 2025/04/29
@version : 1.0.0
@author  : Lucas Cortés.
@contact : lucas.cortes@lanek.cl.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
import random

def clear_page(title='Lanek'):
    try:
        #im = Image.open('assets/logos/favicon.png')
        st.set_page_config(
            page_title=title,
            #page_icon=im,
            layout='wide',
        )
        hide_streamlit_style = """
            <style>
                .reportview-container {
                    margin-top: -2em;
                }
                #MainMenu {visibility: hidden;}
                .stDeployButton {display:none;}
                footer {visibility: hidden;}
                #stDecoration {display:none;}
            </style>
        """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    except Exception:
        pass

def plant_model(current_saturation, valve_opening, natural_oxygen_loss=0.1, proportional_absorption=0.5):
    max_saturation = 100
    saturation_change = proportional_absorption * valve_opening - natural_oxygen_loss
    new_saturation = current_saturation + saturation_change
    return max(0, min(max_saturation, new_saturation))

def plant_model_new(current_saturation, valve_opening, time_step=0.1, natural_oxygen_loss_rate=1.0, proportional_absorption_rate=5.0):
    max_saturation = 100
    natural_oxygen_loss = natural_oxygen_loss_rate * time_step/0.1
    proportional_absorption = proportional_absorption_rate * time_step/0.1
    saturation_change = proportional_absorption * valve_opening - natural_oxygen_loss
    new_saturation = current_saturation + saturation_change
    return max(0, min(max_saturation, new_saturation))

class PIDController:
    def __init__(self, kp, ki, kd, output_min=0, output_max=1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0
        self.output_min = output_min
        self.output_max = output_max

    def control(self, error, dt):
        if dt <= 0:
            raise ValueError("dt must be greater than 0")

        proportional = self.kp * error
        self.integral += error * dt
        integral = self.ki * self.integral
        derivative = self.kd * (error - self.previous_error) / dt

        output = proportional + integral + derivative
        if output > self.output_max:
            output = self.output_max
            self.integral -= error * dt
        elif output < self.output_min:
            output = self.output_min
            self.integral -= error * dt

        self.previous_error = error

        return output

def map_range(array, new_min, new_max):
    old_min = np.min(array)
    old_max = np.max(array)
    if old_max == old_min:
        return np.full_like(array, new_min)
    return (array - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

def divide_list(lst, n):
    length = len(lst)
    base_size = length // n
    remainder = length % n
    parts = []
    start = 0
    for i in range(n):
        size = base_size + (1 if i < remainder else 0)
        parts.append(lst[start:start + size])
        start += size
    return parts

def create_mixed(time, time2, initial_saturation, setpoint, segments_types):
    def create_ramp_segment(arr, start_val, end_val):
        return map_range(arr, start_val, end_val)

    def create_step_segment(arr, value):
        return np.full(len(arr), value)

    def create_triangle_segment(length, start_val, peak_val, end_val):
        half = length // 2
        first = np.linspace(start_val, peak_val, half)
        second = np.linspace(peak_val, end_val, length - half)
        return np.concatenate([first, second])

    def create_gaussian_segment(length, start_val, peak_val):
        center = length // 2
        sigma = length / 6
        x = np.arange(length)
        gauss = np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
        return start_val + gauss * (peak_val - start_val)

    def create_pulse_segment(length, low_val, high_val, pulse_width_ratio=0.3):
        pulse_width = int(length * pulse_width_ratio)
        pulse = np.full(length, low_val)
        start_idx = (length - pulse_width) // 2
        pulse[start_idx:start_idx + pulse_width] = high_val
        return pulse

    n = len(segments_types)
    parts = divide_list(time2, n)
    waveform_parts = []

    random.shuffle(segments_types)

    for i, seg_type in enumerate(segments_types):
        arr = np.array(parts[i])
        length = len(arr)

        if seg_type == 'ramp':
            start_val = initial_saturation if i % 2 == 0 else setpoint
            end_val = setpoint if i % 2 == 0 else initial_saturation
            seg_wave = create_ramp_segment(arr, start_val, end_val)

        elif seg_type == 'step':
            val = initial_saturation + (setpoint - initial_saturation) / 2
            seg_wave = create_step_segment(arr, val)

        elif seg_type == 'triangle':
            peak = setpoint if i % 2 == 0 else initial_saturation
            seg_wave = create_triangle_segment(length, initial_saturation, peak, setpoint)

        elif seg_type == 'gaussian':
            seg_wave = create_gaussian_segment(length, initial_saturation, setpoint)

        elif seg_type == 'pulse':
            seg_wave = create_pulse_segment(length, initial_saturation, setpoint)

        else:
            seg_wave = create_step_segment(arr, initial_saturation)

        waveform_parts.append(seg_wave)

    dynamic_waveform = np.concatenate(waveform_parts)
    remainder_length = len(time) - len(dynamic_waveform)
    remainder = np.full(remainder_length, setpoint) if remainder_length > 0 else np.array([])
    full_waveform = np.concatenate([dynamic_waveform, remainder])
    return full_waveform.tolist()


def create_ramp(time, time2, initial_saturation, setpoint, num_ramps=3):
    segment_length = len(time2) // (2 * num_ramps)
    ramp_values = np.linspace(initial_saturation, setpoint, num_ramps + 1)

    waveform = []
    for i in range(num_ramps):
        ramp = np.linspace(ramp_values[i], ramp_values[i + 1], segment_length)
        waveform.extend(ramp)
        constant_part = [ramp_values[i + 1]] * segment_length
        waveform.extend(constant_part)

    remaining_points = len(time2) - len(waveform)
    if remaining_points > 0:
        waveform.extend([setpoint] * remaining_points)

    constant_part_final = [setpoint] * (len(time) - len(waveform))
    return waveform + constant_part_final

def create_step(time, time2, initial_saturation, setpoint, num_steps=3):
    step_values = np.linspace(initial_saturation, setpoint, num_steps + 1)
    step_length = len(time2) // num_steps
    steps = []

    for i in range(num_steps):
        steps.extend([step_values[i]] * step_length)

    remaining_points = len(time2) - len(steps)
    if remaining_points > 0:
        steps.extend([step_values[-2]] * remaining_points)

    constant_part = [setpoint] * (len(time) - len(steps))
    return steps + constant_part

def create_sine(time, time2, initial_saturation, setpoint, freq=4):
    amplitude = (setpoint - initial_saturation) / 2
    offset = initial_saturation #+ amplitude
    sine_wave = offset + amplitude * np.sin(np.linspace(0, freq*np.pi, len(time2)))
    constant_part = [setpoint] * (len(time) - len(time2))
    return sine_wave.tolist() + constant_part

def create_exponential(time, time2, initial_saturation, setpoint):
    exponential_wave = np.linspace(0, 1, len(time2))
    exponential_wave = initial_saturation + (setpoint - initial_saturation) * np.exp(exponential_wave - 1)
    constant_part = [setpoint] * (len(time) - len(time2))
    return exponential_wave.tolist() + constant_part

def create_triangle(time, time2, initial_saturation, setpoint, num_triangles=2):
    triangle_length = len(time2) // num_triangles
    triangles = []
    for _ in range(num_triangles):
        half_length = triangle_length // 2
        first_half = np.linspace(initial_saturation, setpoint, half_length)
        second_half = np.linspace(setpoint, initial_saturation, triangle_length - half_length)
        triangles.append(np.concatenate([first_half, second_half]))

    all_triangles = np.concatenate(triangles)
    constant_part = [setpoint] * (len(time) - len(all_triangles))
    return all_triangles.tolist() + constant_part


def create_sawtooth(time, time2, initial_saturation, setpoint, cycles=3):
    period = len(time2) // cycles
    sawtooth_wave = []
    for i in range(cycles):
        sawtooth_wave.extend(np.linspace(initial_saturation, setpoint, period))
    sawtooth_wave = sawtooth_wave[:len(time2)]
    constant_part = [setpoint] * (len(time) - len(time2))
    return sawtooth_wave + constant_part

def create_gaussian(time, time2, initial_saturation, setpoint):
    center = len(time2) // 2
    sigma = len(time2) / 10
    gaussian = np.exp(-((np.arange(len(time2)) - center) ** 2) / (2 * sigma ** 2))
    gaussian_wave = initial_saturation + gaussian * (setpoint - initial_saturation)
    constant_part = [setpoint] * (len(time) - len(time2))
    return gaussian_wave.tolist() + constant_part

def create_sigmoid(time, time2, initial_saturation, setpoint):
    x = np.linspace(-6, 6, len(time2))
    sigmoid_wave = 1 / (1 + np.exp(-x))
    sigmoid_wave = initial_saturation + sigmoid_wave * (setpoint - initial_saturation)
    constant_part = [setpoint] * (len(time) - len(time2))
    return sigmoid_wave.tolist() + constant_part

def create_piecewise(time, time2, initial_saturation, setpoint):
    sat0 = map_range(time2, initial_saturation, setpoint)
    sat3 = [setpoint]*((len(time)-len(time2)))
    sat = sat0.tolist() + sat3
    return sat

def create_pulse_train(time, time2, initial_saturation, setpoint, num_pulses=5):
    pulse_width = len(time2) // (2 * num_pulses)
    pulse_spacing = len(time2) // num_pulses
    pulse_train = np.full(len(time2), initial_saturation)

    for i in range(num_pulses):
        start = i * pulse_spacing
        end = min(start + pulse_width, len(time2))
        pulse_train[start:end] = setpoint

    constant_part = [setpoint] * (len(time) - len(time2))
    return pulse_train.tolist() + constant_part


def create_wave(time, time2, initial_saturation, setpoint, waveform, freq):
    if waveform == "Ramps":
        sat = create_ramp(time, time2, initial_saturation, setpoint, freq)
    elif waveform == "Steps":
        sat = create_step(time, time2, initial_saturation, setpoint, freq)
    elif waveform == "Mixed":
        segments = ['ramp', 'step', 'triangle', 'gaussian', 'pulse']
        sat = create_mixed(time, time2, initial_saturation, setpoint, segments)
    elif waveform == "Sine":
        sat = create_sine(time, time2, initial_saturation, setpoint, freq*2)
    elif waveform == "Exponential":
        sat = create_exponential(time, time2, initial_saturation, setpoint)
    elif waveform == "Triangle":
        sat = create_triangle(time, time2, initial_saturation, setpoint, freq)
    elif waveform == "Sawtooth":
        sat = create_sawtooth(time, time2, initial_saturation, setpoint, freq+1)
    elif waveform == "Gaussian":
        sat = create_gaussian(time, time2, initial_saturation, setpoint)
    elif waveform == "Sigmoid":
        sat = create_sigmoid(time, time2, initial_saturation, setpoint)
    elif waveform == "Piecewise":
        sat = create_piecewise(time, time2, initial_saturation, setpoint)
    elif waveform == "Pulse train":
        sat = create_pulse_train(time, time2, initial_saturation, setpoint, freq)
    else:
        sat = [setpoint]*len(time2)
    return sat



def main():
    clear_page('Teve-UCI')
    st.markdown("# Simulador controlador PID")
    waveforms = [
        "Sigmoid",
        "Gaussian",
        "Sine",
        "Triangle",
        "Sawtooth",
        "Pulse train",
        "Ramps",
        "Steps",
        "Mixed",
        "Exponential",
        "Piecewise",
        "Straight",
    ]

    paramWaves = [
        "Sine",
        "Triangle",
        "Sawtooth",
        "Pulse train",
        "Steps",
        "Ramps"
    ]
    try:
        with st.sidebar:
            st.write("Parámetros de simulación")
            waveform = st.selectbox("Forma de onda de la referencia", (waveforms))
            freq = 3
            if waveform in paramWaves:
                freq = st.number_input("Cantidad de repeticiones", 1, 10, freq, 1)

            timespan = st.number_input("Tiempo de referencia", 10, 100, 60, 10)
            setpoint = st.number_input("Saturación deseada", 80, 100, 95, 1)
            initial_saturation = st.number_input("Saturación inicial", 80, 100, 90, 1)
            simulation_time = st.number_input("Tiempo de simulación", 5, 300, 100, 1)
            time_step = st.number_input("Paso de simulación", 0.0, 10.0, 0.1, 0.1)

            st.write("Parámetros del controlador")
            kp = st.number_input("KP", 0.0, 10.0, 2.0, 0.1)
            ki = st.number_input("KI", 0.0, 10.0, 0.5, 0.1)
            kd = st.number_input("KD", 0.0, 10.0, 0.0, 0.01)

            st.write("Parámetros de la planta")

            pl = st.number_input("Absorción proporcional de oxígeno", 0.0, 1.0, 0.5, 0.01)
            nl = st.number_input("Pérdida natural de oxígeno %", 0.0, 1.0, 0.2, 0.1)
            #nl = nlp * time_step
            #st.write(nl)

        pid = PIDController(kp, ki, kd)
        current_saturation = initial_saturation
        valve_opening = 0
        time = np.arange(0, simulation_time, time_step)
        time2 = np.arange(0, timespan, time_step)
        sat = create_wave(time, time2, initial_saturation, setpoint, waveform, freq)

        saturation_values = []
        valve_opening_values = []

        errors = []

        with st.spinner("Simulando...", show_time=True):
            cont = 0
            for t in time:
                if t in time2:
                    error = sat[cont] - current_saturation
                    cont = cont+1
                else:
                    error = setpoint - current_saturation
                valve_opening = pid.control(error, time_step)
                current_saturation = plant_model_new(current_saturation, valve_opening, time_step, nl, pl)
                saturation_values.append(current_saturation)
                valve_opening_values.append(valve_opening)
                errors.append(error)

        # Plot Oxygen Saturation
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=time, y=saturation_values, mode='lines', name='Oxygen Saturation (%)', line=dict(color='blue')
        ))
        fig1.add_trace(go.Scatter(
            x=time, y=sat, name='Reference (%)', line=dict(color='red', dash='dash')
        ))
        #fig1.add_hline(y=setpoint, line=dict(color='red', dash='dash'), name="Setpoint")
        #fig1.add_trace(go.Scatter(
        #    x=time, y=sat3, mode='dash', name='Setpoint (%)', line=dict(color='red')
        #))
        fig1.update_layout(
            title="Oxygen Saturation Control",
            xaxis_title="Time (s)",
            yaxis_title="Oxygen Saturation (%)",
            legend=dict(x=0, y=1),
        )

        # Plot Valve Opening
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=time, y=valve_opening_values, mode='lines', name='Actuation (%)', line=dict(color='green')
        ))
        fig2.update_layout(
            title="Actuation",
            xaxis_title="Time (s)",
            yaxis_title="Valve Opening (0-1)",
            legend=dict(x=0, y=1),
        )

        # Plot Valve Opening
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=time, y=errors, mode='lines', name='Error [-]', line=dict(color='purple')
        ))
        fig3.update_layout(
            title="Error",
            xaxis_title="Time (s)",
            yaxis_title="Error [Flow]",
            legend=dict(x=0, y=1),
        )

        # Render Plots
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)
    except:
        st.error("Error en simulación")




if __name__ == '__main__':
    main()
