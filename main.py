"""
GHOST-1: TERMINAL AGENT (v5.0 Structure/Local)
"""

import argparse
import time
import json
import os
import logging
from colorama import Fore, Style, init

from vision.structure import UIReader
from actions.controller import ActionController

# Try to import Engine, but it's optional for Teacher Mode
try:
    from core.engine import GhostEngine
except ImportError:
    GhostEngine = None

init(autoreset=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_training_data(ui_tree: str, goal: str, action: dict):
    """Log valid supervision data."""
    entry = {
        "ui": ui_tree,
        "goal": goal,
        "output": action
    }
    with open("dataset.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Ghost-1 Terminal Agent")
    parser.add_argument("--teacher", action="store_true", help="Run in Teacher Mode (Data Collection)")
    parser.add_argument("--model", type=str, help="Path to .gguf model")
    args = parser.parse_args()

    print(f"{Fore.CYAN}╔══════════════════════════════════════╗")
    print(f"{Fore.CYAN}║    GHOST-1: LOCAL STRUCTURE (v5.0)   ║")
    print(f"{Fore.CYAN}╚══════════════════════════════════════╝{Style.RESET_ALL}")

    # 1. Initialize Body
    print(f"{Fore.YELLOW}[*] Initializing Eyes (UIReader)...")
    eyes = UIReader()
    hands = ActionController(eyes)

    # 2. Initialize Brain (if not teacher)
    brain = None
    if not args.teacher:
        if GhostEngine:
            try:
                brain = GhostEngine(model_path=args.model)
            except Exception as e:
                print(f"{Fore.RED}[!] Brain failed to load: {e}")
                print(f"{Fore.YELLOW}Falling back to Teacher Mode.")
                args.teacher = True
        else:
            print(f"{Fore.RED}[!] Engine module missing. Falling back to Teacher Mode.")
            args.teacher = True

    # 3. Main Loop
    while True:
        try:
            # New Task
            goal = input(f"\n{Fore.GREEN}User Goal: {Style.RESET_ALL}").strip()
            if not goal: continue
            if goal.lower() in ["exit", "q"]: break

            print(f"{Fore.BLUE}Task: {goal}{Style.RESET_ALL}")
            
            step_count = 0
            while True:
                step_count += 1
                print(f"\n{Fore.YELLOW}--- Step {step_count} ---{Style.RESET_ALL}")
                
                # A. OBSERVE
                print("[*] Scanning UI...")
                ui_tree = eyes.capture_tree()
                # Show simpler tree summary
                lines = ui_tree.split("\n")
                print(f"I see {len(lines)} elements. Top 5:")
                for line in lines[:5]: print(f"  {line}")

                action = None
                
                # B. THINK
                if args.teacher:
                    # Teacher Mode: Ask Human
                    print(f"\n{Fore.MAGENTA}[TEACHER] What should I do?{Style.RESET_ALL}")
                    print("Format: 'click <id>', 'type <id> <text>', 'wait', 'done'")
                    cmd = input("> ").strip().split(" ", 1)
                    
                    op = cmd[0].lower()
                    
                    if op == "click" and len(cmd) > 1:
                        action = {"action": "click", "id": int(cmd[1]), "reason": "Teacher demo"}
                    elif op == "type" and len(cmd) > 1:
                        # Parse "type 5 hello world" -> id=5, text="hello world"
                        parts = cmd[1].split(" ", 1)
                        if len(parts) == 2:
                             action = {"action": "type", "id": int(parts[0]), "text": parts[1], "reason": "Teacher demo"}
                    elif op == "wait":
                         action = {"action": "wait", "reason": "Teacher demo"}
                    elif op == "done":
                         action = {"action": "done", "reason": "Teacher demo"}
                    elif op == "skip":
                         print("Skipping step.")
                         continue
                    else:
                        print("Invalid command. Try again.")
                        continue
                        
                    # Save Data
                    if action:
                        save_training_data(ui_tree, goal, action)
                        print(f"{Fore.MAGENTA}[Recorded to dataset.jsonl]{Style.RESET_ALL}")

                else:
                    # AI Mode
                    print("[*] Thinking...")
                    action = brain.think(ui_tree, goal)
                    print(f"{Fore.CYAN}Brain says: {action}{Style.RESET_ALL}")

                # C. ACT
                if action:
                    result = hands.execute(action)
                    print(f"{Fore.GREEN}>> {result}{Style.RESET_ALL}")
                    
                    if action.get("action") == "done":
                        break
                
                # Safety break
                if step_count > 20:
                    print("Max steps reached.")
                    break
                    
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
