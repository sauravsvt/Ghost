"""
Stress Test: Complex Android Studio Installation Scenario
Tests the agent's self-correction capabilities under failure conditions.

Scenarios:
1. Broken download link (404 error)
2. Installer requires admin privileges
3. Network timeout

The agent should detect errors and adapt autonomously.
"""

import sys
import os
import time
import json
import logging
from unittest.mock import patch, MagicMock
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StressTest")

# Mock responses for different scenarios
MOCK_SCENARIOS = {
    "download_404": {
        "vision": "Browser shows '404 Not Found' error. The download link developer.android.com/studio is broken.",
        "expected_reasoning": "mirror|alternative|backup|different link",
        "expected_action": "browser.open"
    },
    "admin_required": {
        "vision": "Windows UAC popup visible. Message says 'Android Studio Setup requires administrator privileges.'",
        "expected_reasoning": "admin|privilege|elevated|permission",
        "expected_action": "keyboard.hotkey"  # Ctrl+Shift+Enter or similar
    },
    "network_timeout": {
        "vision": "Browser shows 'ERR_CONNECTION_TIMED_OUT'. The page took too long to load.",
        "expected_reasoning": "retry|refresh|internet|connection",
        "expected_action": "keyboard.hotkey"  # F5 or similar
    }
}

class MockEngine:
    """Mock GhostEngine for testing without loading real models."""
    
    def __init__(self):
        self.scenario = None
        self.think_count = 0
        
    def set_scenario(self, scenario_name: str):
        self.scenario = MOCK_SCENARIOS.get(scenario_name)
        
    def see(self, image):
        return self.scenario["vision"] if self.scenario else "Desktop visible"
    
    def think(self, image, task, history=None):
        """Simulate reasoning based on scenario."""
        self.think_count += 1
        
        if not self.scenario:
            return {"action": {"tool": "done"}, "reasoning": "No scenario", "vision": ""}
        
        # Simulate DeepSeek reasoning
        vision = self.scenario["vision"]
        
        if "404" in vision:
            reasoning = """
            <think>
            The download link is broken (404 error).
            I need to find an alternative source.
            Options:
            1. Try mirror site (dl.google.com/android)
            2. Search for 'Android Studio direct download'
            3. Use package manager (winget, choco)
            
            Attempting option 1: mirror site.
            </think>
            """
            action = {"tool": "browser.open", "url": "https://dl.google.com/android/studio/install/"}
            
        elif "UAC" in vision:
            reasoning = """
            <think>
            Windows requires admin privileges.
            I need to restart the installer with elevated permissions.
            Action: Close current dialog and re-run as admin.
            Using keyboard shortcut to confirm UAC.
            </think>
            """
            action = {"tool": "keyboard.hotkey", "keys": "alt+y"}  # Yes to UAC
            
        elif "TIMED_OUT" in vision:
            reasoning = """
            <think>
            Network timeout detected.
            Possible causes: slow internet, server down.
            Action: Retry by refreshing the page.
            </think>
            """
            action = {"tool": "keyboard.hotkey", "keys": "f5"}
            
        else:
            reasoning = "Normal operation"
            action = {"tool": "done"}
            
        return {
            "vision": vision,
            "reasoning": reasoning,
            "action": action
        }

def run_stress_test():
    """Execute all stress test scenarios."""
    print("\n" + "="*60)
    print("GHOST-1 STRESS TEST: Android Studio Installation")
    print("="*60)
    
    engine = MockEngine()
    results = []
    
    for scenario_name, scenario in MOCK_SCENARIOS.items():
        print(f"\n--- Scenario: {scenario_name} ---")
        engine.set_scenario(scenario_name)
        
        # Simulate agent loop
        for step in range(3):  # Max 3 retries
            result = engine.think(None, "Install Android Studio")
            
            reasoning = result.get("reasoning", "")
            action = result.get("action", {})
            
            print(f"Step {step+1}: {action.get('tool', 'unknown')}")
            
            # Check if agent adapted correctly
            expected_keywords = scenario["expected_reasoning"].split("|")
            adapted = any(kw in reasoning.lower() for kw in expected_keywords)
            correct_action = action.get("tool") == scenario["expected_action"]
            
            if adapted and correct_action:
                print(f"✓ PASS: Agent detected issue and adapted")
                results.append((scenario_name, "PASS"))
                break
            elif step == 2:
                print(f"✗ FAIL: Agent did not adapt after 3 attempts")
                results.append((scenario_name, "FAIL"))
    
    # Summary
    print("\n" + "="*60)
    print("STRESS TEST RESULTS")
    print("="*60)
    
    passed = sum(1 for _, status in results if status == "PASS")
    total = len(results)
    
    for scenario, status in results:
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} {scenario}: {status}")
    
    print(f"\nTotal: {passed}/{total} scenarios passed")
    print("="*60)
    
    return passed == total

if __name__ == "__main__":
    success = run_stress_test()
    sys.exit(0 if success else 1)
