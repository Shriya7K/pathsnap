"""Launch script for PathSnap FastAPI Backend & UI Server."""
import os
import sys
import socket
from pathlib import Path
import uvicorn

# Add project root and set cwd
root = Path(__file__).resolve().parent
os.chdir(root)
if str(root) not in sys.path:
    sys.path.insert(0, str(root))


def get_local_ip() -> str:
    """Detect local LAN IPv4 address for multi-device access."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Does not send actual data; routes through active network adapter
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


if __name__ == "__main__":
    port = 8000
    host = "0.0.0.0"  # Bind to all interfaces so other laptops can connect
    local_ip = get_local_ip()

    print("=" * 68)
    print("        PATHSNAP GPS MAP MATCHING SERVER (MULTI-LAPTOP READY)")
    print("=" * 68)
    print(f"  Local Laptop Access:      http://127.0.0.1:{port}/")
    print(f"  Second Laptop / Network:  http://{local_ip}:{port}/")
    print(f"  Interactive Swagger Docs: http://{local_ip}:{port}/docs")
    print("=" * 68)
    print(f"Starting server on 0.0.0.0:{port}...")
    uvicorn.run("api.app:app", host=host, port=port, reload=False, log_level="info")
