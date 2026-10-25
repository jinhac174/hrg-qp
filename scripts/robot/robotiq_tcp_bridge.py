"""Expose a serial port over TCP: the Robotiq 2F-85 stays plugged into the
DROID NUC, run_hardware.py on the workstation talks to it over the network.

On the NUC (Python 3.5+, pyserial):

    python3 robotiq_tcp_bridge.py                     # /dev/ttyUSB0 -> TCP 5020

On the workstation, in configs/robot/hardware.yaml:

    gripper:
      port: socket://192.168.0.127:5020

One client at a time; a new client replaces the old one. Bytes are passed
through unchanged (Modbus RTU frames, request / reply).
"""

import argparse
import select
import socket

import serial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--port", type=int, default=5020)
    args = ap.parse_args()

    ser = serial.Serial(args.device, args.baud, bytesize=8, parity="N", stopbits=1, timeout=0)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port))
    srv.listen(1)
    print("[bridge] {} @ {} <-> tcp :{}".format(args.device, args.baud, args.port), flush=True)

    client = None
    try:
        while True:
            fds = [srv, ser] + ([client] if client else [])
            readable, _, _ = select.select(fds, [], [], 1.0)
            if srv in readable:
                new, addr = srv.accept()
                new.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                if client:
                    client.close()
                client = new
                ser.reset_input_buffer()
                print("[bridge] client {}".format(addr), flush=True)
            if ser in readable:
                data = ser.read(ser.in_waiting or 1)
                if client and data:
                    try:
                        client.sendall(data)
                    except OSError as e:
                        print("[bridge] client lost: {}".format(e), flush=True)
                        client.close()
                        client = None
            if client and client in readable:
                try:
                    data = client.recv(4096)
                except OSError as e:  # reset by peer: the client was killed
                    print("[bridge] client lost: {}".format(e), flush=True)
                    data = b""
                if not data:
                    print("[bridge] client closed", flush=True)
                    client.close()
                    client = None
                else:
                    ser.write(data)
    except KeyboardInterrupt:
        pass
    finally:
        if client:
            client.close()
        srv.close()
        ser.close()


if __name__ == "__main__":
    main()
