"""
Verify Universal Intent on an unknown app (Clock).
"""
import time
import subprocess
from core.engine import GhostEngine
from vision.structure import UIReader

import logging

# Configure Logging to stdout
logging.basicConfig(level=logging.INFO)

def test_universal_intent():
    print("--- TEST: UNIVERSAL INTENT (CLOCK) ---")
    
    # 1. Launch a stranger (Clock)
    print("[*] Launching Clock...")
    subprocess.Popen("start ms-clock:", shell=True)
    time.sleep(3)
    
    # 2. Capture
    reader = UIReader()
    tree = reader.capture_tree()
    
    # 3. Analyze
    brain = GhostEngine()
    print("[?] Asking Brain to analyze UI...")
    
    analysis = brain.analyze_ui(tree)
    
    print("\n[RESULT]")
    print(f"Full Analysis Object: {analysis}")
    print(f"App Type: {analysis.get('app_type')}")
    print(f"Goals: {analysis.get('suggested_goals')}")
    
    if analysis.get("suggested_goals"):
        print("\n✅ SUCCESS: Brain generated goals for new app.")
    else:
        print("\n❌ FAILURE: No goals generated.")

if __name__ == "__main__":
    test_universal_intent()
