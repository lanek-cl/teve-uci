# -*- coding: utf-8 -*-
"""
@file    : main.py
@brief   : Runs main controller simulation
@date    : 2025/04/29
@version : 1.0.0
@author  : Lucas Cortés.
@contact : lucas.cortes@lanek.cl.
"""

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

        # add_logo("assets/logos/ap75.png", height=75)

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

        # with open("./source/style.css") as f:
        #    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass

def plant_model(current_saturation, valve_opening, natural_oxygen_loss=0.01, proportional_absortion=0.5):
    max_saturation = 100
    saturation_change = proportional_absortion * valve_opening - natural_oxygen_loss
    new_saturation = current_saturation + saturation_change
    return max(0, min(max_saturation, new_saturation))

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


def main():
    clear_page('Simulador')
    st.markdown("# Simulador controlador PID")
    try:
        #with st.form("my_form"):
        with st.sidebar:
            st.write("Parámetros de simulación")
            setpoint = st.number_input("Saturación deseada", 80, 100, 95, 1)
            initial_saturation = st.number_input("Saturación inicial", 80, 100, 90, 1)
            simulation_time = st.number_input("Tiempo de simulación", 5, 100, 10, 1)
            time_step = st.number_input("Paso de simulación", 0.0, 10.0, 0.1, 0.1)

            st.write("Parámetros del controlador")
            kp = st.number_input("KP", 0.0, 10.0, 1.0, 0.1)
            ki = st.number_input("KI", 0.0, 1.0, 0.1, 0.1)
            kd = st.number_input("KD", 0.0, 1.0, 0.05, 0.05)

            st.write("Parámetros de la planta")
            pl = st.number_input("Absorción proporcional de oxígeno", 0.0, 1.0, 0.5, 0.1)
            nl = st.number_input("Pérdida natural de oxígeno", 0.0, 1.0, 0.01, 0.01)

        #submitted = st.button('Simular', type='primary') #st.form_submit_button("Submit")
        #if submitted:
        pid = PIDController(kp, ki, kd)
        current_saturation = initial_saturation
        valve_opening = 0
        time = np.arange(0, simulation_time, time_step)
        saturation_values = []
        valve_opening_values = []

        with st.spinner("Simulando...", show_time=True):
            for t in time:
                error = setpoint - current_saturation
                valve_opening = pid.control(error, time_step)
                current_saturation = plant_model(current_saturation, valve_opening, nl, pl)
                saturation_values.append(current_saturation)
                valve_opening_values.append(valve_opening)

        # Plot Oxygen Saturation
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=time, y=saturation_values, mode='lines', name='Oxygen Saturation (%)', line=dict(color='blue')
        ))
        fig1.add_hline(y=setpoint, line=dict(color='red', dash='dash'), name="Setpoint")
        fig1.update_layout(
            title="Oxygen Saturation Control",
            xaxis_title="Time (s)",
            yaxis_title="Oxygen Saturation (%)",
            legend=dict(x=0, y=1),
        )

        # Plot Valve Opening
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=time, y=valve_opening_values, mode='lines', name='Valve Opening (%)', line=dict(color='green')
        ))
        fig2.update_layout(
            title="Valve Opening",
            xaxis_title="Time (s)",
            yaxis_title="Valve Opening (0-1)",
            legend=dict(x=0, y=1),
        )

        # Render Plots
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
    except:
        st.error("Error en simulación")




if __name__ == '__main__':
    main()
