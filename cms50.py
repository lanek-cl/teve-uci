import os
import time
import hid
import csv
from datetime import datetime
import numpy as np

stop_data_collection = False


def connect_device(VENDOR_ID, PRODUCT_ID, retries=100, delay=2):
    for attempt in range(retries):
        try:
            device = hid.device()
            device.open(VENDOR_ID, PRODUCT_ID)
            print("Device connected.")
            return device
        except Exception as e:
            print(f"Connection failed (attempt {attempt + 1}/{retries}): {e}")
            time.sleep(delay * (attempt + 1))  # Exponential backoff
    raise RuntimeError("Failed to connect to device after several attempts.")

def collect_data(device, VENDOR_ID, PRODUCT_ID, csvFileName):
    global stop_data_collection
    data_count = 0
    HR_bit = 0
    SPO2_bit = 0
    PPG_bit = 0
    last_time = time.time()

    with open(csvFileName, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Count', 'PPG', 'HR', 'RR', 'SPO2', "TimeStamp", "DateTime"])

    while not stop_data_collection:
        try:
            data = device.read(18)
            if not data:
                raise OSError("Device might be disconnected.")

            current_time = time.time()
            elapsed_time = current_time - last_time
            timestamps = [current_time - elapsed_time + i * (elapsed_time / 3) for i in range(3)]
            last_time = current_time

            for i in range(3):
                data_update_bit = data[1 + 6 * i]
                PPG_bit = data[3 + 6 * i] if data_update_bit == 0 else PPG_bit
                HR_bit = data[3 + 6 * i] if data_update_bit == 1 else HR_bit
                SPO2_bit = data[4 + 6 * i] if data_update_bit == 1 else SPO2_bit
                RR_bit = 60000/HR_bit if HR_bit != 0 else np.nan

                with open(csvFileName, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([data_count, PPG_bit, HR_bit, RR_bit, SPO2_bit, timestamps[i], datetime.fromtimestamp(timestamps[i]).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]])
                    data_count += 1


        except Exception as e:
            try:
                device.close()
            except:
                pass
            time.sleep(2)
            device = connect_device(VENDOR_ID, PRODUCT_ID)


if __name__ == '__main__':
    name = "raw_data" #input("Ingrese nombre del archivo: ")
    VENDOR_ID = 0x28E9
    PRODUCT_ID = 0x028A
    csvPath = 'csv/'
    os.makedirs(csvPath, exist_ok=True)
    csvFileName = csvPath + f'{name}.csv'

    try:
        device = connect_device(VENDOR_ID, PRODUCT_ID)
        print("Opening the device")
        collect_data(device, VENDOR_ID, PRODUCT_ID, csvFileName)
    except RuntimeError as e:
        print(f"Fatal error: {e}")
