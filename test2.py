import hid
import time

# Define all the known CMS50E commands
commands = {
    "cmd_hello1": b'\x7d\x81\xa7\x80\x80\x80\x80\x80\x80',
    "cmd_hello2": b'\x7d\x81\xa2\x80\x80\x80\x80\x80\x80',
    "cmd_hello3": b'\x7d\x81\xa0\x80\x80\x80\x80\x80\x80',
    "cmd_session_hello": b'\x7d\x81\xad\x80\x80\x80\x80\x80\x80',
    "cmd_get_session_count": b'\x7d\x81\xa3\x80\x80\x80\x80\x80\x80',
    "cmd_get_session_time": b'\x7d\x81\xa5\x80\x80\x80\x80\x80\x80',
    "cmd_get_session_duration": b'\x7d\x81\xa4\x80\x80\x80\x80\x80\x80',
    "cmd_get_user_info": b'\x7d\x81\xab\x80\x80\x80\x80\x80\x80',
    "cmd_get_session_data": b'\x7d\x81\xa6\x80\x80\x80\x80\x80\x80',
    "cmd_get_deviceid": b'\x7d\x81\xaa\x80\x80\x80\x80\x80\x80',
    "cmd_get_info": b'\x7d\x81\xb0\x80\x80\x80\x80\x80\x80',
    "cmd_get_model": b'\x7d\x81\xa8\x80\x80\x80\x80\x80\x80',
    "cmd_get_vendor": b'\x7d\x81\xa9\x80\x80\x80\x80\x80\x80',
    "cmd_session_erase": b'\x7d\x81\xae\x80\x80\x80\x80\x80\x80',
    "cmd_custom": b'\x7d\x81\xf5\x80\x80\x80\x80\x80\x80',
    "cmd_session_stuff": b'\x7d\x81\xaf\x80\x80\x80\x80\x80\x80',
    "cmd_get_live_data": b'\x7d\x81\xa1\x80\x80\x80\x80\x80\x80',
}

def main():
    device = hid.device()
    try:
        #device.open(0x28e9, 0x028a)
        device.open(0x28E9, 0x028A)
        device.set_nonblocking(1)
        print("Device connected.")

        for name, cmd in commands.items():
            print(f"\nSending {name}: {cmd.hex()}")
            device.write(cmd)
            time.sleep(0.1)

            response = device.read(18)
            if response:
                ascii_part = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in response)
                print(f"Response: {ascii_part}")
            else:
                print("No response")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        device.close()
        print("Device closed.")

if __name__ == "__main__":
    main()
