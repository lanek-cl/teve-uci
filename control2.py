# -*- coding: utf-8 -*-
"""
@file    : main.py
@brief   : Runs main controller simulation
@date    : 2025/04/29
@version : 1.0.0
@author  : Lucas Cortés
@contact : lucas.cortes@lanek.cl
"""

import streamlit as st
import plotly.graph_objects as go
import polars as pl
import pandas as pd
import time
import os
from datetime import datetime
import json
import serial
import serial.tools.list_ports
import csv
import numpy as np
import io
import altair as alt

def clear_page(title="Lanek"):
    st.set_page_config(page_title=title, layout="wide")

RAW_DATA = 'csv/raw_data.csv'
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
        proportional = self.kp * error
        self.integral += error * dt
        derivative = self.kd * (error - self.previous_error) / dt

        output = proportional + self.ki * self.integral + derivative
        output = max(self.output_min, min(output, self.output_max))

        if output == self.output_max or output == self.output_min:
            self.integral -= error * dt  # anti-windup

        self.previous_error = error
        return output


def get_sat(max_retries=5, delay=0.1):
    for attempt in range(max_retries):
        try:
            with open(RAW_DATA, "rb") as f:
                f.seek(0, 2)  # EOF
                filesize = f.tell()

                pos = max(filesize - 2, 0)
                f.seek(pos)

                while True:
                    byte = f.read(1)
                    if byte == b'\n' or pos == 0:
                        break
                    pos = max(pos - 2, 0)
                    f.seek(pos)

                last_line = f.readline().decode().strip()

            with open(RAW_DATA, "r", newline='') as f:
                header = next(f).strip().split(",")

            if not last_line or len(last_line.split(",")) != len(header):
                raise ValueError("Last line is empty or malformed")

            values = last_line.split(",")
            row_dict = dict(zip(header, values))

            spo2 = float(row_dict["SPO2"])
            hr = float(row_dict["HR"])
            ppg = float(row_dict["PPG"])
            ts = row_dict["TimeStamp"]
            count = float(row_dict["Count"])
            return min(spo2, 100), ts, hr, ppg, count

        except (OSError, ValueError, KeyError):
            time.sleep(delay)  # wait a bit before retrying

    # If all retries fail, return NaNs and empty timestamp
    return np.nan, "", np.nan, np.nan, np.nan


def get_sat_old():
    df = pl.read_csv("csv/raw_data.csv")
    spo2 = df[-1, "SPO2"]
    ts = df[-1, "TimeStamp"]
    hr = df[-1, "HR"]
    ppg = df[-1, "PPG"]
    count = df[-1, "Count"]
    return min(spo2, 100), ts, hr, ppg, count

def get_sat_prob():
    with open(RAW_DATA, "rb") as f:
        f.seek(-2, 2)  # Move to second last byte
        while f.read(1) != b'\n':
            f.seek(-2, 1)
        last_line = f.readline().decode().strip()

    # Re-read the header to map column names
    with open(RAW_DATA, "r", newline='') as f:
        header = next(f).strip().split(",")

    # Skip if last_line is empty or malformed
    if not last_line or len(last_line.split(",")) != len(header):
        raise ValueError("Last line is empty or malformed")

    # Build a dictionary manually (avoiding DictReader on single line)
    values = last_line.split(",")
    row_dict = dict(zip(header, values))

    spo2 = float(row_dict["SPO2"])
    hr = float(row_dict["HR"])
    ppg = float(row_dict["PPG"])
    ts = row_dict["TimeStamp"]
    count = float(row_dict["Count"])
    return min(spo2, 100), ts, hr, ppg, count


def get_sat_new():
    with open("csv/raw_data.csv", "rb") as f:
        try:
            f.seek(-500, 2)  # Go to near the end of the file (500 bytes before EOF)
        except OSError:
            f.seek(0)  # File is smaller than 500 bytes
        lines = f.readlines()
        last_line = lines[-1].decode()
    
    # Now parse the line manually or with Polars
    df = pl.read_csv(io.StringIO(last_line), has_header=False, new_columns=["TimeStamp", "SPO2", "HR", "PPG", "Count"])
    spo2 = df[0, "SPO2"]
    ts = df[0, "TimeStamp"]
    hr = df[0, "HR"]
    ppg = df[0, "PPG"]
    count = df[0, "Count"]
    return min(spo2, 100), ts, hr, ppg, count

def export_data_to_csv():
    df = pd.DataFrame({
        "timestamp": st.session_state.timestamps,
        "saturation": st.session_state.saturation_values,
        "reference": st.session_state.reference,
        "valve_opening": st.session_state.valve_opening_values,
        "error": st.session_state.errors,
        "hr": st.session_state.hr_values,
        "ppg": st.session_state.ppg_values,
    })
    os.makedirs("output_data", exist_ok=True)
    filename = datetime.now().strftime("simulacion_pid_%Y%m%d_%H%M%S.csv")
    filepath = os.path.join("output_data", filename)
    df.to_csv(filepath, index=False)
    return filepath

def plot():
    if st.session_state.plot == "altair":
        plot_altair()
    if st.session_state.plot == "plotly":
        plot_plotly()
    if st.session_state.plot == "streamlit":
        plot_streamlit()

def plot_plotly():
    satPC = []
    for i in st.session_state.valve_opening_values:
        satPC.append(i*100)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=st.session_state.timestamps, y=st.session_state.reference, name="Referencia", line=dict(color="red", dash="dash")))
    fig1.add_trace(go.Scatter(x=st.session_state.timestamps, y=st.session_state.saturation_values, name="Saturación", line=dict(color="blue")))
    fig1.update_layout(title="Saturación de Oxígeno", xaxis_title="Tiempo (s)", yaxis_title="SpO₂ (%)")
    st.session_state.placeholder1.plotly_chart(fig1, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=st.session_state.timestamps, y=satPC, name="Apertura válvula", line=dict(color="green")))
    fig2.update_layout(title="Apertura", xaxis_title="Tiempo (s)", yaxis_title="Apertura (%)")
    st.session_state.placeholder2.plotly_chart(fig2, use_container_width=True)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=st.session_state.timestamps, y=st.session_state.errors, name="Error", line=dict(color="purple")))
    fig3.update_layout(title="Error", xaxis_title="Tiempo (s)", yaxis_title="Error (%)")
    st.session_state.placeholder3.plotly_chart(fig3, use_container_width=True)

def plot_altair():
    if not st.session_state.timestamps:
        return

    # Calculate valve opening in percent
    satPC = [v * 100 for v in st.session_state.valve_opening_values]

    # === 1. Saturación & Referencia ===
    ts = pd.to_datetime(st.session_state.timestamps)

    # 1. Saturación & Referencia (with dashed red line for Referencia)
    df_saturation = pd.DataFrame({
        "Tiempo": ts,
        "Saturación": st.session_state.saturation_values,
        "Referencia": st.session_state.reference
    })

    df_sat_long = df_saturation.melt(id_vars=["Tiempo"], var_name="Tipo", value_name="Valor")

    chart1 = alt.Chart(df_sat_long).mark_line().encode(
        x=alt.X("Tiempo:T", title="Tiempo"),
        y=alt.Y("Valor:Q", title="SpO₂ (%)", scale=alt.Scale(domain=[min(st.session_state.saturation_values), 100])),
        color=alt.Color("Tipo:N", scale=alt.Scale(domain=["Saturación", "Referencia"], range=["blue", "red"])),
        strokeDash=alt.condition(
            alt.datum.Tipo == "Referencia",
            alt.value([5, 5]),  # dashed for "Referencia"
            alt.value([1])       # solid for "Saturación"
        )
    ).properties(
        title="Saturación de Oxígeno",
        height=300
    )

    st.session_state.placeholder1.altair_chart(chart1, use_container_width=True)

    # === 2. Apertura ===
    df_valve = pd.DataFrame({
        "Tiempo": ts,
        "Apertura (%)": satPC
    })

    chart2 = alt.Chart(df_valve).mark_line(color="green").encode(
        x=alt.X("Tiempo:T", title="Tiempo"),
        y=alt.Y("Apertura (%):Q", title="Apertura (%)", scale=alt.Scale(domain=[min(satPC), 100]))
    ).properties(
        title="Apertura",
        height=300
    )

    st.session_state.placeholder2.altair_chart(chart2, use_container_width=True)

    # === 3. Error ===
    df_error = pd.DataFrame({
        "Tiempo": ts,
        "Error": st.session_state.errors
    })

    chart3 = alt.Chart(df_error).mark_line(color="purple").encode(
        x=alt.X("Tiempo:T", title="Tiempo"),
        y=alt.Y("Error:Q", title="Error (%)", scale=alt.Scale(domain=[min(st.session_state.errors), max(st.session_state.errors)]))
    ).properties(
        title="Error",
        height=300
    )

    st.session_state.placeholder3.altair_chart(chart3, use_container_width=True)


def plot_streamlit():
    if not st.session_state.timestamps:
        return

    # Format time (just HH:MM:SS) for axis labeling if needed
    ts = [t.split(" ")[1] for t in st.session_state.timestamps]

    # === 1. Saturación de Oxígeno ===
    df1 = pd.DataFrame({
        "Saturación": st.session_state.saturation_values,
        "Referencia": st.session_state.reference,
    }, index=ts)
    st.session_state.placeholder1.line_chart(df1)

    # === 2. Apertura válvula (%)
    valve_pct = np.array(st.session_state.valve_opening_values) * 100
    df2 = pd.DataFrame({
        "Apertura válvula": valve_pct,
    }, index=ts)
    st.session_state.placeholder2.line_chart(df2)

    # === 3. Error
    df3 = pd.DataFrame({
        "Error": st.session_state.errors,
    }, index=ts)
    st.session_state.placeholder3.line_chart(df3)

def load_config():
    with open("config.json", "r") as f:
        return json.load(f)


def save_config():
    config = load_config()

    updates = {
        "setpoint": st.session_state.setpoint,
        "time_step": st.session_state.time_step,
        "simulation_time": st.session_state.simulation_time,
        "controller_type": st.session_state.controllerType,
        "kp": st.session_state.kp,
        "ki": st.session_state.ki,
        "kd": st.session_state.kd,
        "port": st.session_state.portNumber,
    }

    for key, value in updates.items():
        if key in config and isinstance(config[key], dict):
            config[key]["default"] = value
        else:
            config[key] = {"default": value}  # fallback in case structure doesn't exist

    with open("config.json", "w") as f:
        json.dump(config, f, indent=4)


def set_open(value):
    arduino = st.session_state.get("arduino", None)
    if arduino and arduino.is_open and 0 <= value <= 255:
        arduino.write(f"{value}\n".encode("utf-8"))


def start():
    st.session_state.update({
        "running": True,
        "start_time": time.time(),
        "timestamps": [],
        "saturation_values": [],
        "hr_values": [],
        "ppg_values": [],
        "valve_opening_values": [],
        "errors": [],
        "reference": [],
        "disabled": True
    })
    st.rerun()


def stop():
    st.session_state["running"] = False
    st.session_state["disabled"] = False
    set_open(0)
    export_data_to_csv()
    st.rerun()


def set_session():
    defaults = {
        "disabled": False,
        "timestamps": [],
        "saturation_values": [],
        "hr_values": [],
        "ppg_values": [],
        "valve_opening_values": [],
        "errors": [],
        "reference": [],
        "running": False,
        "setpoint": 95.0,
        "lastTS": None,
        "plot": "altair",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    st.session_state.placeholder0 = st.empty()
    st.session_state.placeholder1 = st.empty()
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.placeholder2 = st.empty()
    with col2:
        st.session_state.placeholder3 = st.empty()

def run_controller():
    pid = PIDController(st.session_state.kp, st.session_state.ki, st.session_state.kd, st.session_state.time_step)
    start_time = time.time()
    MAX_LOST = 10
    LOST = 0
    while time.time() - start_time <= st.session_state.simulation_time:
        time.sleep(st.session_state.time_step)

        # Retry logic for stale data
        attempts = 3
        for _ in range(attempts):
            current_saturation, current_timestamp, hr, ppg, count = get_sat()
            if count != st.session_state.lastTS:
                break
            time.sleep(0.005)

        if count != st.session_state.lastTS:
            error = st.session_state.setpoint - current_saturation
            valve_opening = pid.control(error, st.session_state.time_step)
            set_open(int(valve_opening * 255))

            # Append real data
            st.session_state.timestamps.append(current_timestamp)
            st.session_state.saturation_values.append(current_saturation)
            st.session_state.hr_values.append(hr)
            st.session_state.ppg_values.append(ppg)
            st.session_state.valve_opening_values.append(valve_opening)
            st.session_state.errors.append(error)
            st.session_state.lastTS = count
            LOST = 0
        else:
            # Append NaNs if data is stale
            current_time = time.time()
            formatted_time = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

            st.session_state.timestamps.append(formatted_time)
            st.session_state.saturation_values.append(np.nan)
            st.session_state.hr_values.append(np.nan)
            st.session_state.ppg_values.append(np.nan)
            st.session_state.valve_opening_values.append(np.nan)
            st.session_state.errors.append(np.nan)
            LOST += 1

        # Reference updated in both branches
        st.session_state.reference = [st.session_state.setpoint] * len(st.session_state.timestamps)

        # Status handling
        if LOST > MAX_LOST:
            st.session_state.placeholder0.error("Data stream stopped, check device.")
        else:
            st.session_state.placeholder0.empty()

        plot()


def get_params():
    st.write("Parámetros de simulación")
    config = load_config()

    def get_input(cfg, typ="number"):
        return st.number_input(
            cfg["label"], cfg["min"], cfg["max"], cfg["default"], cfg["step"],
            disabled=st.session_state.disabled
        ) if typ == "number" else st.selectbox(
            cfg["label"], cfg["options"], index=cfg["options"].index(cfg["default"]),
            disabled=st.session_state.disabled
        )

    st.session_state.setpoint = get_input(config["setpoint"])
    st.session_state.time_step = get_input(config["time_step"])
    st.session_state.simulation_time = get_input(config["simulation_time"])
    st.session_state.controllerType = get_input(config["controller_type"], "select")

    if "P" in st.session_state.controllerType:
        st.session_state.kp = get_input(config["kp"])
    else:
        st.session_state.kp = 0

    if "I" in st.session_state.controllerType:
        st.session_state.ki = get_input(config["ki"])
    else:
        st.session_state.ki = 0

    if "D" in st.session_state.controllerType:
        st.session_state.kd = get_input(config["kd"])
    else:
        st.session_state.kd = 0

    ports = [port.device for port in serial.tools.list_ports.comports()]
    st.session_state.portNumber = st.selectbox(
        config["port"]["label"], ports, index=len(ports) - 1,
        disabled=st.session_state.disabled
    )

    st.session_state.plot = st.selectbox(
        "Plot backend", ["altair", "plotly", "streamlit"],
        disabled=st.session_state.disabled
    )




def main():
    clear_page("Teve-UCI")
    set_session()
    st.sidebar.markdown("# Controlador PID")

    with st.sidebar:
        get_params()
        if "arduino" not in st.session_state or not st.session_state.arduino.is_open:
            try:
                st.session_state.arduino = serial.Serial(st.session_state.portNumber, 9600, timeout=1)
                time.sleep(2)
            except Exception as e:
                st.error(f"No se pudo abrir el puerto {st.session_state.portNumber}: {e}")
                return

        if not st.session_state.running:
            if st.button("START", type="primary"):
                save_config()
                start()
        else:
            if st.button("STOP", type="secondary"):
                stop()

    if st.session_state.running:
        run_controller()
        stop()
    else:
        plot()


if __name__ == "__main__":
    main()
