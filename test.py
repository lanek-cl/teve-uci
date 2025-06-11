import hid
import time

device = hid.device()
device.open(0x28E9, 0x028A)
device.set_nonblocking(1)

device.write(b'\x7d\x81\xa7\x80\x80\x80\x80\x80\x80')  # hello1
time.sleep(0.1)
response = device.read(18)
print(response)

# Send command to start streaming
# device.write(b'\x1b\x00\x10\x30\xdd\x3d\x8e\xbd\xff')
# device.write(b'\xff\x00\x00\x00\x00\x09\x00\x00\x01')
# device.write(b'\x00\x05\x00\x81\x01\x00\x00\x00\x00')
# device.write(b'\x7d\x81\xa1\x80\x80\x80\x80\x80\x80')
# time.sleep(0.1)
# data = device.read(18)
# if data:
#     ascii_part = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in data)
#     print(f"ASCII: {ascii_part}")
#     device.write(b'\x1b\x00\x10\x30\xdd\x3d\x8e\xbd\xff')
#     device.write(b'\xff\x00\x00\x00\x00\x09\x00\x00\x01')
#     device.write(b'\x00\x05\x00\x81\x01\x00\x00\x00\x00')
#     time.sleep(0.1)
#     data = device.read(18)
#     if data:
#         ascii_part = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in data)
#         print(f"ASCII: {ascii_part}")

print("Started stream. Listening for data...")

while True:
    data = device.read(18)
    if data:
        # Filter printable ASCII characters (32-126) from data
        ascii_part = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in data)
        print(f"Raw: {data}")
        print(f"ASCII: {ascii_part}")
    time.sleep(0.05)
