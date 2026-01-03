"""
GHOST-1: TERMINAL AGENT (v5.0 Structure/Local)
"""

import argparse
import time
import json
import os
from datetime import datetime
from colorama import Fore, Style, init

from vision.structure import UIReader
from actions.controller import ActionController
from core.logging_utils import setup_logging, get_logger

# Try to import Engine, but it's optional for Teacher Mode
try:
    from core.engine import GhostEngine
except ImportError:
    GhostEngine = None

init(autoreset=True)

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
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging (DEBUG level)")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set logging level")
    parser.add_argument("--json-logs", action="store_true", help="Enable JSON formatted logs")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum steps per task (default: 20)")
    parser.add_argument("--version", action="version", version="Ghost-1 v5.0 (Structure/Local)")
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else args.log_level
    setup_logging(log_level=log_level, enable_json=args.json_logs)
    logger = get_logger("Main")
    
    logger.info("Ghost-1 starting...")
    logger.debug(f"Arguments: {vars(args)}")

    print(f"{Fore.CYAN}╔══════════════════════════════════════╗")
    print(f"{Fore.CYAN}║    GHOST-1: LOCAL STRUCTURE (v5.0)   ║")
    print(f"{Fore.CYAN}╚══════════════════════════════════════╝{Style.RESET_ALL}")

    # 1. Initialize Body
    print(f"{Fore.YELLOW}[*] Initializing Eyes (UIReader)...")
    logger.debug("Initializing UIReader")
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
            action_history = []  # Track recent actions to detect loops
            previous_tree = ""
            stuck_count = 0  # Count how many times screen didn't change
            
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
                    # Check for infinite loops (same action repeated 3+ times)
                    action_signature = f"{action.get('action')}_{action.get('id', '')}"
                    action_history.append(action_signature)
                    
                    if len(action_history) >= 3 and action_history[-3:].count(action_signature) == 3:
                        print(f"{Fore.RED}[!] Loop detected! Same action repeated 3 times: {action_signature}{Style.RESET_ALL}")
                        print(f"{Fore.YELLOW}Trying scroll or wait instead...{Style.RESET_ALL}")
                        # Fallback: try scrolling to find new elements
                        hands.execute({"action": "scroll.down"})
                        action_history = []  # Reset history
                        stuck_count += 1
                        if stuck_count > 2:
                            print(f"{Fore.RED}Agent is stuck. Stopping.{Style.RESET_ALL}")
                            break
                        continue
                    
                    # Keep only last 5 actions
                    if len(action_history) > 5:
                        action_history.pop(0)
                    
                    result = hands.execute(action)
                    print(f"{Fore.GREEN}>> {result}{Style.RESET_ALL}")
                    
                    # D. VERIFY: Did the screen change?
                    time.sleep(0.5)  # Brief wait for UI to update
                    new_tree = eyes.capture_tree()
                    
                    if new_tree == previous_tree and action.get("action") not in ["wait", "done"]:
                        stuck_count += 1
                        print(f"{Fore.YELLOW}[!] Screen didn't change (stuck: {stuck_count}/3){Style.RESET_ALL}")
                        if stuck_count >= 3:
                            print(f"{Fore.RED}[!] No progress for 3 steps. Stopping.{Style.RESET_ALL}")
                            break
                    else:
                        stuck_count = 0  # Reset if screen changed
                    
                    previous_tree = ui_tree  # Update reference
                    
                    if action.get("action") == "done":
                        break
                
                # Safety break
                if step_count > args.max_steps:
                    print(f"Max steps ({args.max_steps}) reached.")
                    logger.warning(f"Max steps reached: {args.max_steps}")
                    break
                    
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
