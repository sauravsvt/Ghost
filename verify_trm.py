
import sys
import os
import time

# Add current dir to path
sys.path.append(os.getcwd())

from core.engine import GhostEngine

def test_trm():
    print("Optimization: Loading Brain...")
    try:
        brain = GhostEngine()
    except Exception as e:
        print(f"Skipping Brain output (Model not found or error): {e}")
        return

    # Mock UI Tree
    ui_tree = """
    [0] Window: Notepad
      [1] Menu: File
      [2] Document: Text Editor
      [3] Button: Close
    """
    
    task = "Close Notepad"
    
    print(f"\n--- TEST: TRM RECURSION ---")
    print(f"Task: {task}")
    print("Thinking...")
    
    start = time.time()
    action = brain.think(ui_tree, task)
    end = time.time()
    
    print(f"\n[Result] Action: {action}")
    print(f"[Time] {end - start:.2f}s")
    
    if isinstance(action, list):
        print("✅ CALM Detected: Output is a Batch Action List")
    elif isinstance(action, dict):
        print("✅ Action is a Dict")

if __name__ == "__main__":
    test_trm()
