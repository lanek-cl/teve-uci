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
import csv
import numpy as np
import io
import altair as alt

def clear_page(title="Lanek"):
    st.set_page_config(page_title=title, layout="wide")


RAW_DATA = "csv/raw_data.csv"

def get_sat():
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
    if st.session_state.timestamps and st.session_state.saturation_values:
        ts = st.session_state.timestamps

        df_saturation = pd.DataFrame({
            "Tiempo": ts,
            "Saturación": st.session_state.saturation_values,
            "Referencia": st.session_state.reference
        })

        df1_long = df_saturation.melt(id_vars=["Tiempo"], var_name="Tipo", value_name="Valor")

        saturacion = alt.Chart(df1_long[df1_long["Tipo"] == "Saturación"]).mark_line(color="blue").encode(
            x=alt.X("Tiempo:T", title="Tiempo"),
            y=alt.Y("Valor:Q", title="SpO₂ (%)", scale=alt.Scale(domain=[min(st.session_state.saturation_values), 100]))
        )

        # Referencia line (dashed red)
        referencia = alt.Chart(df1_long[df1_long["Tipo"] == "Referencia"]).mark_line(color="red", strokeDash=[5,5]).encode(
            x=alt.X("Tiempo:T"),
            y=alt.Y("Valor:Q")
        )

        chart1 = saturacion + referencia
        chart1 = chart1.properties(
            title="Saturación de Oxígeno",
            height=300
        )

        st.session_state.placeholder1.altair_chart(chart1, use_container_width=True)



def load_config():
    with open("config.json", "r") as f:
        return json.load(f)


def save_config():
    config = load_config()

    updates = {
        "setpoint": st.session_state.setpoint,
        "time_step": st.session_state.time_step,
        "simulation_time": st.session_state.simulation_time,
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
            valve_opening = 0
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

   

def main():
    clear_page("Teve-UCI")
    set_session()
    st.sidebar.markdown("# Controlador PID")

    with st.sidebar:
        get_params()
        
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
