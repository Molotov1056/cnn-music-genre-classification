#!/usr/bin/env python3

import subprocess
import sys
import time
import webbrowser

def find_available_port():
    """Find an available port starting from 8503"""
    import socket
    
    for port in range(8503, 8510):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    return 8503  # fallback

def main():
    port = find_available_port()
    print(f"🚀 Starting Music Genre Classifier on port {port}")
    print(f"🌐 Open your browser to: http://localhost:{port}")
    print("📱 To stop the app, press Ctrl+C\n")
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", str(port),
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down the web app. Goodbye!")

if __name__ == "__main__":
    main()