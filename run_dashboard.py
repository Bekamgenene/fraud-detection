"""
Launch script for the fraud detection dashboard.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit dashboard."""
    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
    
    print("🚀 Starting Fraud Detection Dashboard...")
    print("📊 Dashboard will open in your browser")
    print("🛑 Press Ctrl+C to stop the dashboard")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port=8501",
            "--server.address=localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        print("💡 Make sure you have installed all dependencies: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
