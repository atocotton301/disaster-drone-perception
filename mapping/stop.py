"""Request a clean stop of this host's running disaster system."""
import socket


if __name__ == '__main__':
    try:
        with socket.create_connection(('127.0.0.1', 18764), timeout=3) as client:
            client.sendall(b'STOP\n')
            print(client.recv(128).decode())
    except OSError as e:
        raise SystemExit('No running disaster system, or stop endpoint unavailable: '+str(e))
