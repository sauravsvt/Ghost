"""
Verify Brain Logic Only (No App Launch).
"""
import logging
from core.engine import GhostEngine

# Configure Logging
logging.basicConfig(level=logging.INFO)

def test_brain_intent():
    print("--- TEST: BRAIN INTENT (MOCK UI) ---")
    
    # 1. Mock UI Tree (Clock-like)
    mock_tree = """
    [Window] Clock
      [Button] Stop Watch (ID 10)
      [Button] Alarm (ID 11)
      [Button] Timer (ID 12)
      [Text] 12:00 PM
      [Button] Add New Alarm (ID 20)
    """
    
    # 2. Analyze
    brain = GhostEngine()
    print("[?] Asking Brain to analyze Mock UI...")
    
    analysis = brain.analyze_ui(mock_tree)
    
    print("\n[RESULT]")
    print(f"Full Analysis Object: {analysis}")
    print(f"App Type: {analysis.get('app_type')}")
    print(f"Goals: {analysis.get('suggested_goals')}")

if __name__ == "__main__":
    test_brain_intent()
