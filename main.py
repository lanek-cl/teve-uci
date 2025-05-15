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

def plant_model(current_saturation, valve_opening, natural_oxygen_loss=0.02, proportional_absorption=0.5):
    max_saturation = 100
    saturation_change = proportional_absorption * valve_opening - natural_oxygen_loss
    new_saturation = current_saturation + saturation_change
    return max(0, min(max_saturation, new_saturation))

class PIDController:
    def __init__(self, kp, ki, kd, output_min=0, output_max=1):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.previous_error = 0
        self.integral = 0
        self.output_min = output_min  # Minimum output limit
        self.output_max = output_max  # Maximum output limit

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
    return (array - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

def divide_list(lst):
    # Calculate the size of each part
    n = len(lst)
    part_size = n // 3
    remainder = n % 3  # For distributing extra elements evenly

    # Create the parts
    parts = []
    start = 0
    for i in range(3):
        extra = 1 if i < remainder else 0  # Add an extra element to some parts if there's a remainder
        end = start + part_size + extra
        parts.append(lst[start:end])
        start = end

    return parts

def create_ramp(time2, initial_saturation, setpoint):
    middle = initial_saturation + (setpoint - initial_saturation)/2
    parts = divide_list(time2)
    sat0 = map_range(parts[0], initial_saturation, middle)
    sat1 = map_range(parts[1], middle, middle)
    sat2 = map_range(parts[2], middle, setpoint)
    sat = sat0.tolist()[:-1] + sat1.tolist() + sat2.tolist()[1:] + [setpoint]*2
    return sat

def create_step(time2, initial_saturation, setpoint):
    middle = initial_saturation + (setpoint - initial_saturation)/2
    parts = divide_list(time2)
    sat0 = map_range(parts[0], initial_saturation, initial_saturation)
    sat1 = map_range(parts[1], middle, middle)
    sat2 = map_range(parts[2], setpoint, setpoint)
    sat = sat0.tolist()[:-1] + sat1.tolist() + sat2.tolist()[1:] + [setpoint]*2
    return sat

def create_mixed(time2, initial_saturation, setpoint):
    middle = initial_saturation + (setpoint - initial_saturation)/2
    parts = divide_list(time2)
    sat0 = map_range(parts[0], initial_saturation, initial_saturation)
    sat1 = map_range(parts[1], middle, middle)
    sat2 = map_range(parts[2], middle, setpoint)
    sat = sat0.tolist()[:-1] + sat1.tolist() + sat2.tolist()[1:] + [setpoint]*2
    return sat

def main():
    clear_page('Simulador')
    st.markdown("# Simulador controlador PID")
    try:
        with st.sidebar:
            st.write("Parámetros de simulación")
            waveform = st.selectbox("Forma de referencia", ["Straight", "Ramps", "Steps", "Mixed"])
            timespan = st.number_input("Tiempo de referencia", 10, 100, 60, 10)
            setpoint = st.number_input("Saturación deseada", 80, 100, 95, 1)
            initial_saturation = st.number_input("Saturación inicial", 80, 100, 90, 1)
            simulation_time = st.number_input("Tiempo de simulación", 5, 300, 100, 1)
            time_step = st.number_input("Paso de simulación", 0.0, 10.0, 1.0, 0.1)

            st.write("Parámetros del controlador")
            kp = st.number_input("KP", 0.0, 10.0, 2.0, 0.1)
            ki = st.number_input("KI", 0.0, 10.0, 0.5, 0.1)
            kd = st.number_input("KD", 0.0, 10.0, 0.0, 0.01)

            st.write("Parámetros de la planta")
            pl = st.number_input("Absorción proporcional de oxígeno", 0.0, 1.0, 0.5, 0.01)
            nl = st.number_input("Pérdida natural de oxígeno", 0.0, 1.0, 0.1, 0.01)

        pid = PIDController(kp, ki, kd)
        current_saturation = initial_saturation
        valve_opening = 0
        time = np.arange(0, simulation_time, time_step)
        time2 = np.arange(0, timespan, time_step)
        if waveform == "Ramps":
            sat = create_ramp(time2, initial_saturation, setpoint)
        elif waveform == "Steps":
            sat = create_step(time2, initial_saturation, setpoint)
        elif waveform == "Mixed":
            sat = create_mixed(time2, initial_saturation, setpoint)
        else:
            sat = [setpoint]*len(time2)
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
                current_saturation = plant_model(current_saturation, valve_opening, nl, pl)
                saturation_values.append(current_saturation)
                valve_opening_values.append(valve_opening)
                errors.append(error)

        # Plot Oxygen Saturation
        if waveform == "Ramps":
            sat2 = [setpoint]*((len(time)-len(time2))+1)
            sat3 = sat[1:]+sat2
        else:
            sat2 = [setpoint]*((len(time)-len(time2)))
            sat3 = sat+sat2
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=time, y=saturation_values, mode='lines', name='Oxygen Saturation (%)', line=dict(color='blue')
        ))
        fig1.add_trace(go.Scatter(
            x=time, y=sat3, name='Reference (%)', line=dict(color='red', dash='dash')
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
