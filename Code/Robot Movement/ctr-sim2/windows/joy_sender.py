import XInput
import socket
import json
import time

WSL_IP = "172.28.133.98"
WSL_PORT = 9870

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rate = 50  # Hz

    print("Xbox controller connected!")
    print(f"Sending to {WSL_IP}:{WSL_PORT} at {rate}Hz")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            state = XInput.get_state(0)

            lx, ly = XInput.get_thumb_values(state)[0]
            rx, ry = XInput.get_thumb_values(state)[1]
            lt, rt = XInput.get_trigger_values(state)

            axes = [lx, ly, lt, rx, ry, rt]

            btns = XInput.get_button_values(state)
            buttons = [
                int(btns.get('A', 0)),
                int(btns.get('B', 0)),
                int(btns.get('X', 0)),
                int(btns.get('Y', 0)),
                int(btns.get('LEFT_SHOULDER', 0)),
                int(btns.get('RIGHT_SHOULDER', 0)),
                int(btns.get('BACK', 0)),
                int(btns.get('START', 0)),
                int(btns.get('LEFT_THUMB', 0)),
                int(btns.get('RIGHT_THUMB', 0)),
            ]

            # Print so you can see it working
            print(f"\rLX:{lx:+.2f} LY:{ly:+.2f} RX:{rx:+.2f} RY:{ry:+.2f} LT:{lt:.2f} RT:{rt:.2f} Btns:{buttons}", end="")

            msg = json.dumps({"axes": axes, "buttons": buttons})
            sock.sendto(msg.encode(), (WSL_IP, WSL_PORT))

            time.sleep(1.0 / rate)

    except XInput.XInputNotConnectedError:
        print("No Xbox controller connected!")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sock.close()

if __name__ == "__main__":
    main()