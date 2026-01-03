"""
GHOST-1 Live Demo: Open Start Menu and Search
Uses Win+S to open search, then navigates
"""
import sys
import time
import pyautogui
from vision.structure import UIReader
from actions.controller import ActionController
from core.engine import GhostEngine

# Redirect to file
log = open("demo_results.txt", "w", encoding="utf-8")
sys.stdout = log
sys.stderr = log

print("=" * 60)
print("🤖 GHOST-1 LIVE DEMO: Open Notepad via Search")
print("=" * 60)

try:
    # Step 0: Open Windows Search
    print("\n[Prep] Opening Windows Search...")
    pyautogui.hotkey('win', 's')  # Open search
    time.sleep(2)
    
    # Step 1: Initialize
    print("\n[1/5] Initializing Ghost-1...")
    eyes = UIReader()
    brain = GhostEngine(model_path="models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
    hands = ActionController(eyes)
    print("✓ All systems ready")
    
    # Step 2: Scan Search Window
    print("\n[2/5] Scanning Windows Search...")
    ui_tree = eyes.capture_tree()
    lines = ui_tree.split('\n')
    print(f"✓ Found {len(lines)} UI elements")
    print("\nSearch window elements (first 20):")
    for i, line in enumerate(lines[:20], 1):
        print(f"  {line}")
    
    # Step 3: Find Search Box
    print("\n[3/5] Finding search box...")
    task = "Type in the search box"
    action = brain.think(ui_tree, task)
    print(f"🧠 Decision: {action}")
    
    # Step 4: Type "notepad"
    print("\n[4/5] Typing 'notepad'...")
    if action.get('action') == 'type':
        # Override the text to type "notepad"
        action['text'] = 'notepad'
    result = hands.execute(action)
    print(f"✓ {result}")
    
    time.sleep(2)  # Wait for search results
    
    # Step 5: Scan results and click Notepad
    print("\n[5/5] Finding and clicking Notepad...")
    ui_tree = eyes.capture_tree()
    lines = ui_tree.split('\n')
    print(f"✓ Found {len(lines)} results")
    print("\nSearch results (first 15):")
    for i, line in enumerate(lines[:15], 1):
        print(f"  {line}")
    
    task = "Click on Notepad app"
    action = brain.think(ui_tree, task)
    print(f"\n🧠 Decision: {action}")
    
    result = hands.execute(action)
    print(f"✓ {result}")
    
    print("\n" + "=" * 60)
    print("🎉 DEMO COMPLETE!")
    print("=" * 60)
    print("\nCheck your screen - did Notepad open?")
    
except Exception as e:
    print(f"\n❌ Demo Failed: {e}")
    import traceback
    traceback.print_exc()

log.close()

# Print to console
with open("demo_results.txt", "r", encoding="utf-8") as f:
    print(f.read())
