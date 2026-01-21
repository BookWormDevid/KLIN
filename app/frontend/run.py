#!/usr/bin/env python3
"""
KLIN Frontend - Launch Script
Run this file to start the web interface
"""

import threading
import webbrowser

"""from app import app"""


def open_browser():
    """Open web browser when server starts"""
    webbrowser.open_new("http://192.168.210.85:5000/")


def main():
    """Main function to start the frontend"""
    print("🚀 Starting KLIN Frontend...")
    print("=" * 50)
    print("📊 System: Video Analysis Interface")
    print("🌐 Server: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)

    # Open browser after short delay
    threading.Timer(1.5, open_browser).start()

    # Start Flask development server
    """ app.run(debug=True, host="192.168.210.85", port=5000, use_reloader=False)"""


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
