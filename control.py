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
import polars as pl
import time
import pandas as pd
import os
from datetime import datetime


def clear_page(title="Lanek"):
    try:
        # im = Image.open('assets/logos/favicon.png')
        st.set_page_config(
            page_title=title,
            # page_icon=im,
            layout="wide",
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
        #st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    except Exception:
        pass



class PIDController:
    def __init__(self, kp, ki, kd, time_step, output_min=0, output_max=1):
        self.kp = kp * 0.1 / time_step
        self.ki = ki * 0.1 / time_step
        self.kd = kd * 0.1 / time_step
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


def get_sat():
    df = pl.read_csv("csv/Output1.csv")
    last_row = df[-1]
    spo2 = last_row["SPO2"][0]
    spo2 = spo2 if spo2 <= 100 else 100
    return spo2

def export_data_to_csv():
    df = pd.DataFrame({
        "timestamp": st.session_state.timestamps,
        "saturation": st.session_state.saturation_values,
        "reference": st.session_state.reference,
        "valve_opening": st.session_state.valve_opening_values,
        "error": st.session_state.errors,
    })
    #return df.to_csv(index=False).encode("utf-8")
    # Create folder if it doesn't exist
    output_dir = "output_data"
    os.makedirs(output_dir, exist_ok=True)

    # Use timestamp in filename
    filename = datetime.now().strftime("simulacion_pid_%Y%m%d_%H%M%S.csv")
    filepath = os.path.join(output_dir, filename)

    df.to_csv(filepath, index=False)
    return filepath

def plot(placeholder1, placeholder2, placeholder3):
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=st.session_state.timestamps, y=st.session_state.saturation_values, mode="lines", name="Saturación", line=dict(color="blue")))
    fig1.add_trace(go.Scatter(x=st.session_state.timestamps, y=st.session_state.reference, mode="lines", name="Referencia", line=dict(color="red", dash="dash")))
    fig1.update_layout(title="Saturación de Oxígeno", xaxis_title="Tiempo (s)", yaxis_title="SpO₂ (%)")
    placeholder1.plotly_chart(fig1, use_container_width=True)

    # Plot 2: Valve opening
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=st.session_state.timestamps, y=st.session_state.valve_opening_values, mode="lines", name="Apertura válvula", line=dict(color="green")))
    fig2.update_layout(title="Actuación", xaxis_title="Tiempo (s)", yaxis_title="Apertura [0–1]")
    placeholder2.plotly_chart(fig2, use_container_width=True)

    # Plot 3: Error
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=st.session_state.timestamps, y=st.session_state.errors, mode="lines", name="Error", line=dict(color="purple")))
    fig3.update_layout(title="Error", xaxis_title="Tiempo (s)", yaxis_title="Error")
    placeholder3.plotly_chart(fig3, use_container_width=True)

def main():
    clear_page("Teve-UCI")
    st.sidebar.markdown("# Controlador PID")
    placeholder1 = st.empty()
    col1, col2 = st.columns(2)
    with col1:
        placeholder2 = st.empty()
    with col2:
        placeholder3 = st.empty()


    if "timestamps" not in st.session_state:
        st.session_state.timestamps = []
    if "saturation_values" not in st.session_state:
        st.session_state.saturation_values = []
    if "valve_opening_values" not in st.session_state:
        st.session_state.valve_opening_values = []
    if "errors" not in st.session_state:
        st.session_state.errors = []
    if "reference" not in st.session_state:
        st.session_state.reference = []
    with st.sidebar:
        st.write("Parámetros de simulación")
        setpoint = st.number_input("Saturación final", 90, 100, 100, 1)
        time_step = st.number_input("Paso de simulación", 0.0, 1.0, 0.1, 0.1)
        simulation_time = st.number_input("Tiempo de simulación", 5, 500, 30, 1)

        if "running" not in st.session_state:
            st.session_state.running = False

        st.write("Parámetros del controlador")
        controllerType = st.selectbox(
            "Tipo de controlador", ["PID", "PI", "PD", "P"]
        )
        kp = (
            st.number_input("KP", 0.0, 4.0, 2.0, 0.1)
            if "P" in controllerType
            else 0
        )
        ki = (
            st.number_input("KI", 0.0, 10.0, 0.5, 0.1)
            if "I" in controllerType
            else 0
        )
        kd = (
            st.number_input("KD", 0.0, 0.1, 0.01, 0.01)
            if "D" in controllerType
            else 0
        )
        if not st.session_state.running:
            if st.button("START", type="primary"):
                st.session_state.running = True
                st.session_state.start_time = time.time()
                st.session_state.timestamps = []
                st.session_state.saturation_values = []
                st.session_state.valve_opening_values = []
                st.session_state.errors = []
                st.session_state.reference = []       
                st.session_state.config = {
                    "setpoint": setpoint,
                    "time_step": time_step,
                    "simulation_time": simulation_time,
                }
                st.rerun()
        else:
            if st.button("STOP", type="secondary"):
                st.session_state.running = False
                export_data_to_csv()
                st.rerun()


    if st.session_state.running:
        try:
            pid = PIDController(kp, ki, kd, time_step)
            valve_opening = 0
            current_saturation = get_sat()
            error = setpoint - current_saturation
            

            start_time = time.time()

            while time.time() - start_time <= simulation_time:
                time.sleep(time_step)
                now = time.time() - start_time
                st.session_state.timestamps.append(now)

                error = setpoint - current_saturation
                valve_opening = pid.control(error, time_step)
                current_saturation = get_sat()

                st.session_state.saturation_values.append(current_saturation)
                st.session_state.valve_opening_values.append(valve_opening)
                st.session_state.errors.append(error)

                # Reference line (same length as time)
                st.session_state.reference = [setpoint] * len(st.session_state.timestamps)
                plot(placeholder1, placeholder2, placeholder3)
            export_data_to_csv()

        except:
            st.error("Error en simulación")

    else:
        plot(placeholder1, placeholder2, placeholder3)


if __name__ == "__main__":
    main()
