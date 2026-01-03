"""
SMART GYM v2.1: Dialog Intelligence + Causal Training
Handles popups, sign-ins, file pickers - truly autonomous learning
"""

import time
import json
import random
import subprocess
import psutil
import pyautogui
import win32gui
import win32con
from colorama import Fore, Style, init
from vision.structure import UIReader
from actions.controller import ActionController

# --- CONFIGURATION ---
TRAINING_CIRCUIT = {
    "notepad.exe": "Notepad",
    "mspaint.exe": "Paint",
    "calc.exe": "Calculator",
    "wordpad.exe": "WordPad",
}

DATASET_FILE = "dataset.jsonl"
STEPS_PER_APP = 50
CYCLES = 5

class SmartGym:
    def __init__(self):
        init()
        self.reader = UIReader()
        self.hands = ActionController(self.reader)
        # Memory of "Bad Buttons" that kill the app
        self.hazards = set()
        # Memory of "Dialog Triggers" (neutral - just dismiss)
        self.dialog_triggers = set()
        self.total_learned = 0

    def detect_context_type(self, tree, app_name):
        """
        🧠 CONTEXT INTELLIGENCE: What situation are we in?
        Returns: "normal", "dialog", "wrong_window", or "window_gone"
        """
        # Case 1: Window completely gone
        if "No active window" in tree or "Error" in tree:
            return "window_gone"
        
        # Case 2: Dialog/Popup appeared
        # Look for dialog indicators
        dialog_keywords = ["Dialog:", "Sign in", "Print", "Save as", "Open file", 
                          "Popup", "Warning", "Alert", "Confirm"]
        if any(keyword in tree for keyword in dialog_keywords):
            return "dialog"
        
        # Case 3: Different app got focus
        top_lines = tree.split("\n")[:5]
        app_in_title = any(app_name.lower() in line.lower() for line in top_lines)
        if not app_in_title:
            return "wrong_window"
        
        # Case 4: Normal state
        return "normal"

    def dismiss_dialog(self):
        """
        🚪 AUTO-DISMISS: Try to close unwanted dialogs
        Strategy: ESC first, then click Cancel/Close if present
        """
        print(f"{Fore.YELLOW}[💬] Dialog detected. Attempting dismissal...{Style.RESET_ALL}")
        
        # Strategy 1: Press Escape (works for most dialogs)
        pyautogui.press('esc')
        time.sleep(0.5)
        
        # Verify dismissal
        tree = self.reader.capture_tree()
        if "Dialog" not in tree and "Sign in" not in tree:
            print(f"{Fore.GREEN}[✓] Dialog dismissed with ESC{Style.RESET_ALL}")
            return True
        
        # Strategy 2: Look for Cancel/Close button
        lines = tree.split("\n")
        for line in lines:
            if "Cancel" in line or "Close" in line:
                try:
                    parts = line.split("]", 1)
                    btn_id = int(parts[0].replace("[", "").strip())
                    self.hands.execute({"action": "click", "id": btn_id})
                    time.sleep(0.5)
                    print(f"{Fore.GREEN}[✓] Dialog dismissed by clicking button{Style.RESET_ALL}")
                    return True
                except:
                    continue
        
        print(f"{Fore.RED}[✗] Could not dismiss dialog{Style.RESET_ALL}")
        return False

    def is_app_focused(self, app_name):
        """Checks if the target app is currently the active window AND visible"""
        tree = self.reader.capture_tree()
        
        # Check if window exists in title
        lines = tree.split("\n")[:5]
        app_in_title = any(app_name.lower() in line.lower() for line in lines)
        
        # Even if app is in title, check if it's actually responding
        # A minimized window will still show in tree but won't be interactable
        if not app_in_title:
            return False
        
        # Check for signs of minimized/hidden state
        # If we see very few elements (<5), window is likely minimized
        element_count = len([l for l in tree.split("\n") if "[" in l and "]" in l])
        if element_count < 5:
            return False
        
        return True

    def launch_app(self, executable, app_name):
        print(f"{Fore.YELLOW}[*] Launching {executable}...{Style.RESET_ALL}")
        subprocess.Popen(executable, shell=True)
        time.sleep(3)
        
        if not self.is_app_focused(app_name):
            print(f"{Fore.RED}[!] Failed to focus {app_name}. Retrying click...{Style.RESET_ALL}")
            pyautogui.click(960, 540)
            time.sleep(1)

    def kill_app(self, executable):
        if executable == "explorer.exe": 
            return
        for proc in psutil.process_iter(['name']):
            try:
                if proc.info['name'] and proc.info['name'].lower() == executable.lower():
                    proc.kill()
            except: 
                pass

    def train(self):
        print(f"{Fore.CYAN}╔══════════════════════════════════════╗")
        print(f"║   SMART GYM v2.1: DIALOG INTELLIGENCE║")
        print(f"║   Handles Popups | Learns from Pain  ║")
        print(f"╚══════════════════════════════════════╝{Style.RESET_ALL}")

        for cycle in range(CYCLES):
            print(f"\n{Fore.MAGENTA}=== CYCLE {cycle+1}/{CYCLES} ==={Style.RESET_ALL}")
            
            for exe, app_name in TRAINING_CIRCUIT.items():
                print(f"\n{Fore.BLUE}>>> Training on: {app_name}{Style.RESET_ALL}")
                self.launch_app(exe, app_name)
                
                consecutive_failures = 0
                
                for i in range(STEPS_PER_APP):
                    try:
                        # 1. READ STATE
                        tree = self.reader.capture_tree()
                        
                        # 2. CONTEXT DETECTION - What's happening?
                        context = self.detect_context_type(tree, app_name)
                        
                        if context == "window_gone":
                            # TRUE HAZARD: Window died
                            print(f"{Fore.RED}[!] Window truly lost! (Minimized/Killed){Style.RESET_ALL}")
                            self.kill_app(exe)
                            time.sleep(1)
                            self.launch_app(exe, app_name)
                            consecutive_failures = 0
                            continue
                        
                        elif context == "dialog":
                            # NEUTRAL: Just a dialog - dismiss and continue
                            if self.dismiss_dialog():
                                continue
                            else:
                                # Failed to dismiss - reset app
                                self.kill_app(exe)
                                self.launch_app(exe, app_name)
                                continue
                        
                        elif context == "wrong_window":
                            # Lost focus - try to refocus by clicking taskbar
                            consecutive_failures += 1
                            if consecutive_failures > 5:
                                print(f"{Fore.RED}[!] Too many focus failures. App likely minimized - restarting{Style.RESET_ALL}")
                                self.kill_app(exe)
                                self.launch_app(exe, app_name)
                                consecutive_failures = 0
                                continue
                            
                            print(f"{Fore.YELLOW}[!] Focus lost. Trying taskbar restore...{Style.RESET_ALL}")
                            # Click bottom of screen (taskbar area) to restore minimized window
                            import win32gui
                            import win32con
                            try:
                                # Find window and restore it
                                hwnd = win32gui.FindWindow(None, app_name)
                                if hwnd:
                                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                                    win32gui.SetForegroundWindow(hwnd)
                                    time.sleep(1)
                                else:
                                    # Fallback: click taskbar
                                    pyautogui.click(960, 1070)  # Taskbar position
                                    time.sleep(0.5)
                            except:
                                pyautogui.click(960, 1070)
                                time.sleep(0.5)
                            continue
                        
                        # 3. PARSE CANDIDATES
                        lines = tree.split("\n")
                        candidates = []
                        
                        for line in lines:
                            if any(x in line for x in ["Button", "MenuItem", "TabItem", "ListItem"]):
                                try:
                                    parts = line.split("]", 1)
                                    id_part = int(parts[0].replace("[", "").strip())
                                    name_part = parts[1].split(":")[1].strip().replace("'", "")
                                    
                                    # AVOID KNOWN HAZARDS
                                    hazard_key = f"{app_name}:{id_part}"
                                    if hazard_key in self.hazards:
                                        continue
                                    
                                    # AVOIDDIALOG TRIGGERS if we saw them before
                                    # (But don't block them entirely - might be useful later)
                                        
                                    # AVOID OBVIOUS KILLERS
                                    if any(bad in name_part.lower() for bad in ["close", "exit", "shutdown", "delete"]):
                                        continue
                                        
                                    candidates.append({"id": id_part, "name": name_part})
                                except: 
                                    continue

                        if not candidates:
                            print("[-] No targets. Waiting...")
                            consecutive_failures += 1
                            if consecutive_failures > 5:
                                print(f"{Fore.YELLOW}[!] Too many failures. Resetting app.{Style.RESET_ALL}")
                                self.kill_app(exe)
                                self.launch_app(exe, app_name)
                                consecutive_failures = 0
                            time.sleep(2)
                            continue

                        # 4. ACT
                        target = random.choice(candidates)
                        print(f"[{i+1}/{STEPS_PER_APP}] Trying: {target['name']} (ID {target['id']})")
                        
                        before_tree = tree
                        self.hands.execute({"action": "click", "id": target['id']})
                        time.sleep(1.0)
                        
                        # 5. OBSERVE CAUSALITY with CONTEXT AWARENESS
                        after_tree = self.reader.capture_tree()
                        after_context = self.detect_context_type(after_tree, app_name)
                        
                        if after_context == "window_gone":
                            # TRUE HAZARD
                            print(f"{Fore.RED}[⚠] HAZARD: '{target['name']}' killed the window!{Style.RESET_ALL}")
                            
                            hazard_key = f"{app_name}:{target['id']}"
                            self.hazards.add(hazard_key)
                            
                            # Save negative example
                            negative_entry = {
                                "ui": before_tree,
                                "goal": f"Click {target['name']}",
                                "output": {"action": "click", "id": target['id']},
                                "result": "hazard",
                                "reason": f"Window disappeared (minimized or closed)"
                            }
                            
                            with open(DATASET_FILE, "a", encoding="utf-8") as f:
                                f.write(json.dumps(negative_entry) + "\n")
                            
                            self.total_learned += 1
                            continue
                        
                        elif after_context == "dialog":
                            # NEUTRAL: Dialog appeared
                            print(f"{Fore.CYAN}[💬] DIALOG: '{target['name']}' opened a popup{Style.RESET_ALL}")
                            
                            dialog_key = f"{app_name}:{target['id']}"
                            self.dialog_triggers.add(dialog_key)
                            
                            # Save neutral example
                            neutral_entry = {
                                "ui": before_tree,
                                "goal": f"Click {target['name']}",
                                "output": {"action": "click", "id": target['id']},
                                "result": "dialog",
                                "reason": f"Opened a dialog/popup (neutral - can dismiss)"
                            }
                            
                            with open(DATASET_FILE, "a", encoding="utf-8") as f:
                                f.write(json.dumps(neutral_entry) + "\n")
                            
                            self.total_learned += 1
                            
                            # Dismiss and continue
                            self.dismiss_dialog()
                            continue
                        
                        # POSITIVE: State changed normally
                        elif before_tree != after_tree:
                            print(f"{Fore.GREEN}[+] SUCCESS: '{target['name']}' -> ID {target['id']}{Style.RESET_ALL}")
                            
                            positive_entry = {
                                "ui": before_tree,
                                "goal": f"Click {target['name']}",
                                "output": {"action": "click", "id": target['id'], "reason": f"User wants to click {target['name']}"},
                                "result": "positive"
                            }
                            
                            with open(DATASET_FILE, "a", encoding="utf-8") as f:
                                f.write(json.dumps(positive_entry) + "\n")
                            
                            self.total_learned += 1
                            consecutive_failures = 0
                        else:
                            print(f"{Fore.LIGHTBLACK_EX}[-] No effect.{Style.RESET_ALL}")

                    except KeyboardInterrupt:
                        print("\n[!] Interrupted by user")
                        self.kill_app(exe)
                        return
                    except Exception as e:
                        print(f"[!] Error: {e}")
                        continue
                
                # Cleanup
                self.kill_app(exe)
                time.sleep(2)

        print(f"\n{Fore.GREEN}🎉 Smart Training Complete!{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Patterns learned: {self.total_learned}{Style.RESET_ALL}")
        print(f"{Fore.RED}Hazards blocked: {len(self.hazards)}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Dialog triggers: {len(self.dialog_triggers)}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}Dataset: {DATASET_FILE}{Style.RESET_ALL}")

if __name__ == "__main__":
    gym = SmartGym()
    gym.train()
