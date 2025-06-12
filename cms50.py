import time
import hid
import csv
from datetime import datetime

# Delay the start time of data collection
def delay_start(delay_time):
    print("Start recording in...")
    for i in range(delay_time):
        print(delay_time - i)
        time.sleep(1)

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
    data_count = 0
    start_time = time.time()
    HR_bit = 0
    SPO2_bit = 0
    PPG_bit = 0

    with open(csvFileName, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Count', 'PPG', 'HR', 'SPO2', "TimeStamp"])

    while True:
        try:
            data = device.read(18)
            if not data:
                print("No data received. Reconnecting...")
                raise OSError("Device might be disconnected.")

            #print(data)
            current_time = time.time()
            formatted_time = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            time_count = current_time - start_time

            for i in range(3):
                check_bit = data[0 + 6 * i]
                data_update_bit = data[1 + 6 * i]
                status_bit = data[2 + 6 * i]

                PPG_bit = data[3 + 6 * i] if data_update_bit == 0 else PPG_bit
                HR_bit = data[3 + 6 * i] if data_update_bit == 1 else HR_bit
                SPO2_bit = data[4 + 6 * i] if data_update_bit == 1 else SPO2_bit

                with open(csvFileName, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([data_count, PPG_bit, HR_bit, SPO2_bit, formatted_time])
                    #print("HR_BPM:", HR_bit, "- SpO2:", SPO2_bit)
                data_count += 1

        except (OSError, Exception, IOError) as e:
            print(f"Error reading from device: {e}")
            print("Attempting to reconnect...")
            try:
                device.close()
            except:
                pass
            time.sleep(2)
            device = connect_device(VENDOR_ID, PRODUCT_ID)

if __name__ == '__main__':
    VENDOR_ID = 0x28E9
    PRODUCT_ID = 0x028A
    csvPath = 'csv/'
    csvFileName = csvPath + 'Output1.csv'

    try:
        device = connect_device(VENDOR_ID, PRODUCT_ID)
        print("Opening the device")
        collect_data(device, VENDOR_ID, PRODUCT_ID, csvFileName)
    except RuntimeError as e:
        print(f"Fatal error: {e}")
