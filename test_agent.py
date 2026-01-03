"""
Quick test of Ghost-1 agent - saves output to file
"""
import sys
from vision.structure import UIReader
from actions.controller import ActionController

# Redirect output
log_file = open("test_results.txt", "w", encoding="utf-8")
sys.stdout = log_file
sys.stderr = log_file

print("=" * 50)
print("GHOST-1 DEBUG TEST")
print("=" * 50)

try:
    # 1. Test UI Reader
    print("\n[1/3] Testing Matrix Reader (UIA)...")
    eyes = UIReader()
    ui_tree = eyes.capture_tree()
    lines = ui_tree.split('\n')
    print(f"✓ Scanned {len(lines)} lines in UI tree")
    print("\nFirst 10 elements:")
    for line in lines[:10]:
        print(f"  {line}")
    
    # 2. Test Brain
    print("\n[2/3] Testing Local Brain (Qwen 0.5B)...")
    from core.engine import GhostEngine
    model_path = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    brain = GhostEngine(model_path=model_path)
    print("✓ Brain loaded successfully")
    
    # 3. Test Decision
    print("\n[3/3] Testing Decision Loop...")
    task = "Open Notepad"
    print(f"\nGoal: '{task}'")
    print(f"UI Context size: {len(ui_tree)} characters")
    print("\nThinking...")
    
    action = brain.think(ui_tree, task)
    print(f"\n📄 Raw Brain Output:")
    print(f"   {action}")
    print(f"\n🧠 Brain Decision:")
    print(f"   Action: {action.get('action')}")
    print(f"   ID: {action.get('id', 'N/A')}")
    print(f"   Reason: {action.get('reason', 'N/A')}")
    
    # 4. Execute
    print("\n[4/4] Executing action...")
    hands = ActionController(eyes)
    result = hands.execute(action)
    print(f"✓ Execution Result: {result}")
    
    print("\n" + "=" * 50)
    print("✅ TEST PASSED - All systems operational!")
    print("=" * 50)
    
except Exception as e:
    print(f"\n❌ TEST FAILED")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

log_file.close()
print("Results saved to test_results.txt")
