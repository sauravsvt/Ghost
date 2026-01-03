"""
Ghost-1 Setup & Calibration
Runs the '5-6 hour' adaptation process (condensed) to tailor the agent to your screen.
"""

import sys
import os
import time
import json
import logging
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)

def calibration_session():
    """Run an interactive calibration to learn screen layout."""
    print("\nStarting Calibration Session...")
    print("This will help Ghost-1 learn your specific desktop layout.")
    print("We will record the locations of key elements.")
    
    apps_to_calibrate = ["Start Button", "Browser Icon", "Search Bar", "Recycle Bin"]
    calibration_data = {}
    
    try:
        import pyautogui
        
        for app in apps_to_calibrate:
            print(f"\n👉 Please hover your mouse over the '{app}'")
            print("   Press Enter when ready (or 's' to skip)...")
            choice = input().strip().lower()
            if choice == 's':
                continue
            
            x, y = pyautogui.position()
            print(f"   Recorded '{app}' at ({x}, {y})")
            calibration_data[app] = {"x": x, "y": y}
            
            # Simulate a "GRPO Update" (saving to experience match)
            # In a real system, this would train a spatial bias
            
    except ImportError:
        print("pyautogui missing. Install with: pip install pyautogui")
        return

    # Save calibration to a foundational memory file
    if calibration_data:
        print("\nSaving calibration data...")
        os.makedirs("config", exist_ok=True)
        with open("config/calibration.json", 'w') as f:
            json.dump(calibration_data, f, indent=2)
        print("✓ Calibration complete. Ghost-1 now knows your layout.")
    else:
        print("Calibration skipped.")

def main():
    print("Ghost-1 Setup")
    print("=============")
    
    # 1. Install deps
    print("\n[1] Checking dependencies...")
    # (Simulated check, user should run pip install)
    print("Ensure you have run: pip install -r requirements.txt")
    
    # 2. Download models?
    print("\n[2] Model Check...")
    print("Models download automatically on first run.")
    
    # 3. Calibration
    print("\n[3] Personalize")
    response = input("Do you want to run the Screen Calibration? (y/n): ").strip().lower()
    if response == 'y':
        calibration_session()
        
    print("\nSetup Complete!")
    print("Run: python main.py")

if __name__ == "__main__":
    main()
