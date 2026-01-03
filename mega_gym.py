"""
GHOST MEGA GYM: Circuit Training
Autonomous unsupervised learning across multiple applications
"""

import time
import json
import random
import subprocess
import psutil
import os
from colorama import Fore, Style, init
from vision.structure import UIReader
from actions.controller import ActionController

# --- CONFIGURATION ---
# The Gym will cycle through these apps automatically
# Format: "executable_name": "Friendly Name"
TRAINING_CIRCUIT = {
    "notepad.exe": "Notepad",
    "mspaint.exe": "Paint",
    "calc.exe": "Calculator",
    "wordpad.exe": "WordPad",
    # "explorer.exe": "File Explorer" # Uncomment if you want to train navigation (won't close)
}

DATASET_FILE = "dataset.jsonl"
STEPS_PER_APP = 50  # Actions to try per app before rotating
CYCLES = 5          # How many times to repeat the full circuit

def launch_app(executable):
    """Launches an app and waits for it to load"""
    print(f"{Fore.YELLOW}[*] Launching {executable}...{Style.RESET_ALL}")
    try:
        # Use Popen to launch without blocking
        subprocess.Popen(executable, shell=True)
        time.sleep(3) # Give it time to render UI
    except Exception as e:
        print(f"[!] Failed to launch {executable}: {e}")

def kill_app(executable):
    """Closes the app to clean up"""
    if executable.lower() == "explorer.exe": 
        return # Safety: Never kill Explorer
    
    found = False
    for proc in psutil.process_iter(['name']):
        try:
            if proc.info['name'] and proc.info['name'].lower() == executable.lower():
                proc.kill()
                found = True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    if found:
        print(f"{Fore.RED}[*] Closed {executable}{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}[!] Could not find process to close: {executable}{Style.RESET_ALL}")

def main():
    init()
    print(f"{Fore.CYAN}╔══════════════════════════════════════╗")
    print(f"║     GHOST MEGA GYM: CIRCUIT TRAINING ║")
    print(f"║     Unsupervised Learning Mode       ║")
    print(f"╚══════════════════════════════════════╝{Style.RESET_ALL}")

    try:
        reader = UIReader()
        hands = ActionController(reader)
    except Exception as e:
        print(f"{Fore.RED}[!] Error initializing tools: {e}{Style.RESET_ALL}")
        return

    total_learned = 0

    for cycle in range(CYCLES):
        print(f"\n{Fore.MAGENTA}=== CYCLE {cycle+1}/{CYCLES} ==={Style.RESET_ALL}")
        
        for exe, app_name in TRAINING_CIRCUIT.items():
            print(f"\n{Fore.BLUE}>>> Training on: {app_name}{Style.RESET_ALL}")
            
            # 1. SETUP: Launch App
            launch_app(exe)
            
            # 2. TRAIN: Explore for N steps
            for i in range(STEPS_PER_APP):
                try:
                    # READ STATE
                    tree = reader.capture_tree()
                    
                    # PARSE CANDIDATES (Find buttons/menus)
                    lines = tree.split("\n")
                    candidates = []
                    for line in lines:
                        # Look for interactable types
                        # Format is typically "[ID] Type: 'Name'"
                        if any(x in line for x in ["Button", "MenuItem", "TabItem", "ListItem", "TreeItem", "Hyperlink"]):
                            try:
                                # Safe Parsing
                                parts = line.split("]", 1)
                                id_part = int(parts[0].replace("[", "").strip())
                                rest = parts[1]
                                name_part = rest.split(":")[1].strip().replace("'", "")
                                
                                # SAFETY FILTER: Skip dangerous keywords
                                if any(bad in name_part.lower() for bad in ["delete", "close", "exit", "shutdown", "format", "remove"]):
                                    continue
                                    
                                candidates.append({"id": id_part, "name": name_part, "line": line})
                            except: 
                                continue

                    if not candidates:
                        print("[-] No interactive elements found. Waiting...")
                        time.sleep(2)
                        continue

                    # PRIORITIZE: Score candidates by usefulness
                    # Buttons > MenuItems > Others
                    for candidate in candidates:
                        score = 0
                        if "Button" in candidate['line']: score += 3
                        if "MenuItem" in candidate['line']: score += 2
                        if "TabItem" in candidate['line']: score += 1
                        # Prefer named elements
                        if candidate['name'] and len(candidate['name']) > 2:
                            score += 1
                        candidate['score'] = score
                    
                    # Sort by score (highest first), then randomize top 10
                    candidates.sort(key=lambda x: x['score'], reverse=True)
                    top_candidates = candidates[:10]  # Take top 10
                    target = random.choice(top_candidates)  # Pick randomly from top
                    
                    print(f"[{i+1}/{STEPS_PER_APP}] Trying: {target['name']} (ID {target['id']}, Score: {target['score']})")
                    
                    # Execute click
                    result = hands.execute({"action": "click", "id": target['id']})
                    
                    # OBSERVE: Wait for reaction
                    time.sleep(1.0)
                    new_tree = reader.capture_tree()
                    
                    # VERIFY: Did the screen change?
                    if tree != new_tree:
                        print(f"{Fore.GREEN}[+] SUCCESS: Learned '{target['name']}' -> ID {target['id']}{Style.RESET_ALL}")
                        
                        # DEDUPLICATION CHECK: Avoid saving duplicate patterns
                        # Check if we already have this exact goal
                        goal_text = f"Click {target['name']}"
                        duplicate = False
                        
                        try:
                            if os.path.exists(DATASET_FILE):
                                with open(DATASET_FILE, "r", encoding="utf-8") as f:
                                    for line in f:
                                        existing = json.loads(line)
                                        if existing.get("goal") == goal_text:
                                            duplicate = True
                                            break
                        except:
                            pass
                        
                        if not duplicate:
                            # SAVE EXPERIENCE
                            entry = {
                                "ui": tree,
                                "goal": goal_text,
                                "output": {"action": "click", "id": target['id'], "reason": f"User wants to click {target['name']}"}
                            }
                            
                            with open(DATASET_FILE, "a", encoding="utf-8") as f:
                                f.write(json.dumps(entry) + "\n")
                            
                            total_learned += 1
                        else:
                            print(f"{Fore.YELLOW}[~] Skipped duplicate pattern{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.LIGHTBLACK_EX}[-] No effect.{Style.RESET_ALL}")
                    
                except KeyboardInterrupt:
                    print("\n[!] Pausing...")
                    kill_app(exe)
                    return
                except Exception as e:
                    print(f"[!] Error: {e}")
                    continue
            
            # 3. CLEANUP: Close App
            kill_app(exe)
            time.sleep(2) # Cooldown

    print(f"\n{Fore.GREEN}🎉 Training Complete. Total new patterns learned: {total_learned}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Dataset saved to: {DATASET_FILE}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
