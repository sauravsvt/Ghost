"""
UNIVERSAL GYM (Ghost v3.0)
True Autonomy: No hardcoded apps. Discovery via LLM.
"""

import time
import json
import psutil
import pyautogui
import win32gui
import win32process
from colorama import Fore, Style, init
from vision.structure import UIReader
from actions.controller import ActionController
from core.engine import GhostEngine

DATASET_FILE = "dataset_universal.jsonl"
CYCLES = 3

class UniversalGym:
    def __init__(self):
        init()
        self.reader = UIReader()
        self.hands = ActionController(self.reader)
        self.brain = GhostEngine()
        self.hazards = set()
        self.total_learned = 0

    def get_active_app_info(self):
        """Get the name/exe of the currently focused window."""
        try:
            hwnd = win32gui.GetForegroundWindow()
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            proc = psutil.Process(pid)
            return proc.name(), win32gui.GetWindowText(hwnd)
        except:
            return "unknown.exe", "Unknown"

    def train(self):
        print(f"{Fore.CYAN}╔══════════════════════════════════════╗")
        print(f"║     UNIVERSAL GYM (Ghost v3.0)       ║")
        print(f"║   Zero-Shot Learning | Semantic Data ║")
        print(f"╚══════════════════════════════════════╝{Style.RESET_ALL}")

        for cycle in range(CYCLES):
            print(f"\n{Fore.MAGENTA}=== CYCLE {cycle+1}/{CYCLES} ==={Style.RESET_ALL}")
            
            # 1. Orient
            exe_name, window_title = self.get_active_app_info()
            print(f"{Fore.BLUE}[*] Focused App: {window_title} ({exe_name}){Style.RESET_ALL}")
            
            if "code" in exe_name.lower() or "terminal" in exe_name.lower(): 
                 print(f"{Fore.RED}[!] Safety: Skipping Code/Terminal to avoid self-recursion.{Style.RESET_ALL}")
                 time.sleep(2)
                 continue

            tree = self.reader.capture_tree()
            
            # 2. Universal Intent Discovery
            print(f"{Fore.YELLOW}[?] Analyzing UI for Intent...{Style.RESET_ALL}")
            analysis = self.brain.analyze_ui(tree)
            
            app_type = analysis.get("app_type", "Unknown")
            goals = analysis.get("suggested_goals", [])
            
            print(f"{Fore.GREEN}[✓] Context: {app_type}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}[✓] Goals: {goals}{Style.RESET_ALL}")
            
            # 3. Dynamic Exploration Loop
            for goal in goals:
                print(f"\n{Fore.CYAN}>>> Goal: {goal}{Style.RESET_ALL}")
                
                # RECURSIVE THINKING
                start_tree = self.reader.capture_tree()
                action = self.brain.think(start_tree, goal)
                
                print(f"    Action: {action}")
                
                # Execute
                result_msg = self.hands.execute(action)
                print(f"    Result: {result_msg}")
                time.sleep(2)
                
                # Verify
                end_tree = self.reader.capture_tree()
                
                if start_tree != end_tree:
                    print(f"{Fore.GREEN}[+] SUCCESS: UI Changed{Style.RESET_ALL}")
                    # Save Semantic Lesson
                    lesson = {
                        "ui": start_tree,
                        "goal": goal,
                        "output": action,
                        "result": "positive",
                        "context": app_type
                    }
                    with open(DATASET_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps(lesson) + "\n")
                    self.total_learned += 1
                else:
                    print(f"{Fore.LIGHTBLACK_EX}[-] No visible change.{Style.RESET_ALL}")
            
            print(f"\n{Fore.YELLOW}[*] Switching context or waiting for user...{Style.RESET_ALL}")
            time.sleep(5)

if __name__ == "__main__":
    gym = UniversalGym()
    gym.train()
