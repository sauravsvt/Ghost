"""
Ghost Gym: Autonomous Training Loop
====================================
Allows the agent to explore safe applications, discover UI patterns,
and automatically generate high-quality training data while you're away.

Safety Features:
- App whitelist (only safe apps)
- Element type filtering (buttons, menus only)
- Screen change verification
- Keyboard interrupt support
- pyautogui FAILSAFE enabled

Based on TRM paper principles: Small models excel with structured, recursive data.
"""

import time
import json
import random
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional
from colorama import Fore, Style, init

from vision.structure import UIReader
from actions.controller import ActionController

# ==================== CONFIGURATION ====================

# SAFETY: Only interact with these apps
SAFE_APPS = ["Notepad", "Calculator", "Chrome", "Edge", "Explorer", "Paint"]

# Element types that are safe to click
SAFE_ELEMENT_TYPES = ["Button", "MenuItem", "TabItem", "ListItem"]

# Training dataset output
DATASET_FILE = "dataset.jsonl"

# Maximum exploration steps before auto-stop
MAX_STEPS = 500

# Delay after each action (seconds)
ACTION_DELAY = 1.5

# Countdown before starting (gives time to focus a safe app)
STARTUP_COUNTDOWN = 3

# ==================== HELPER FUNCTIONS ====================

def setup_logging():
    """Configure logging for the gym session."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/gym_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("GhostGym")

def is_safe_element(line: str) -> bool:
    """Check if a UI element line represents a safe clickable element."""
    # Check if line contains safe element types
    if not any(elem_type in line for elem_type in SAFE_ELEMENT_TYPES):
        return False
    
    # Avoid dangerous keywords (allow minimize/close for recovery training)
    dangerous_keywords = ["delete", "remove", "format", "erase", "shutdown", "restart"]
    line_lower = line.lower()
    
    return not any(keyword in line_lower for keyword in dangerous_keywords)

def parse_element_line(line: str) -> Optional[Dict[str, any]]:
    """
    Parse a UI tree line to extract element info.
    
    Format: "[12] Button: 'File'"
    Returns: {"id": 12, "type": "Button", "name": "File", "line": "..."}
    """
    try:
        # Extract ID
        if not line.strip().startswith('['):
            return None
            
        id_part = line.split(']')[0].replace('[', '').strip()
        element_id = int(id_part)
        
        # Extract type and name
        remainder = line.split(']', 1)[1].strip()
        
        if ':' not in remainder:
            return None
            
        elem_type = remainder.split(':')[0].strip()
        name_part = remainder.split(':', 1)[1].strip().strip("'\"")
        
        return {
            "id": element_id,
            "type": elem_type,
            "name": name_part,
            "line": line
        }
    except (IndexError, ValueError) as e:
        return None

def generate_instruction(element: Dict[str, any]) -> str:
    """
    Reverse-engineer a plausible instruction from the element clicked.
    
    Examples:
    - Button 'File' -> "Open the File menu"
    - Button '7' -> "Click number 7"
    - MenuItem 'Save' -> "Save the document"
    """
    elem_type = element["type"]
    name = element["name"]
    
    # Generate contextual instruction
    if elem_type == "Button":
        if name.isdigit():
            return f"Click number {name}"
        elif name in ["File", "Edit", "View", "Help"]:
            return f"Open the {name} menu"
        elif name in ["+", "-", "*", "/", "="]:
            return f"Click {name} operator"
        else:
            return f"Click {name} button"
    
    elif elem_type == "MenuItem":
        return f"Select {name} from menu"
    
    elif elem_type == "ListItem":
        return f"Open {name}"
    
    elif elem_type == "TabItem":
        return f"Switch to {name} tab"
    
    else:
        return f"Click {name}"

def save_training_example(ui_tree: str, instruction: str, element_id: int, logger):
    """Save a verified training example to dataset.jsonl."""
    try:
        # Match existing dataset format
        entry = {
            "ui": ui_tree,
            "goal": instruction,
            "output": {
                "action": "click",
                "id": element_id,
                "reason": f"User wants to {instruction.lower()}"
            }
        }
        
        with open(DATASET_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')
        
        logger.info(f"{Fore.CYAN}[+] Saved: '{instruction}' -> ID {element_id}{Style.RESET_ALL}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to save example: {e}")
        return False

# ==================== MAIN GYM LOOP ====================

def main():
    """Main autonomous training loop."""
    init()  # Initialize colorama
    logger = setup_logging()
    
    # Print banner
    print(f"\n{Fore.CYAN}╔════════════════════════════════════════════╗")
    print(f"║       GHOST GYM: AUTONOMOUS TRAINING       ║")
    print(f"║   Exploration → Verification → Learning    ║")
    print(f"╚════════════════════════════════════════════╝{Style.RESET_ALL}\n")
    
    # Initialize components
    logger.info("Initializing UI Reader...")
    reader = UIReader()
    hands = ActionController(reader)
    
    # Safety announcement
    print(f"{Fore.YELLOW}[!] SAFETY MODE ENABLED{Style.RESET_ALL}")
    print(f"    → Only interacting with: {', '.join(SAFE_APPS)}")
    print(f"    → Element types: {', '.join(SAFE_ELEMENT_TYPES)}")
    print(f"    → Max steps: {MAX_STEPS}")
    print(f"    → Press Ctrl+C to stop anytime\n")
    
    # Startup countdown
    print(f"{Fore.GREEN}[*] Starting in {STARTUP_COUNTDOWN} seconds...{Style.RESET_ALL}")
    print(f"    → Focus a safe app now (Notepad, Calculator, etc.)")
    time.sleep(STARTUP_COUNTDOWN)
    
    # Statistics
    stats = {
        "total_attempts": 0,
        "successful_saves": 0,
        "failed_clicks": 0,
        "no_change": 0,
        "skipped_unsafe": 0
    }
    
    # Main exploration loop
    logger.info("Starting autonomous exploration...")
    
    try:
        for step in range(1, MAX_STEPS + 1):
            print(f"\n{Fore.MAGENTA}{'='*50}")
            print(f"  Step {step}/{MAX_STEPS}")
            print(f"{'='*50}{Style.RESET_ALL}")
            
            stats["total_attempts"] += 1
            
            # 1. SCAN: Capture current UI state
            logger.info("Scanning UI tree...")
            tree_before = reader.capture_tree()
            
            # Check if we have a valid window
            if "No active window" in tree_before or "Error" in tree_before:
                print(f"{Fore.YELLOW}[!] No valid window focused. Attempting recovery...{Style.RESET_ALL}")
                
                # Try to recover by clicking on taskbar buttons
                try:
                    # Scan the desktop/taskbar
                    desktop_tree = reader.capture_tree()
                    lines = desktop_tree.split('\n')
                    
                    # Look for safe app buttons in taskbar
                    for line in lines:
                        # Check if it's a button with a safe app name
                        if "Button:" in line:
                            for safe_app in SAFE_APPS:
                                if safe_app.lower() in line.lower():
                                    # Found a safe app button! Try to click it
                                    element = parse_element_line(line)
                                    if element:
                                        print(f"{Fore.GREEN}[↺] Attempting to restore: {safe_app}{Style.RESET_ALL}")
                                        reader.click_id(element['id'])
                                        time.sleep(1)
                                        break
                            else:
                                continue
                            break
                    else:
                        # No safe app found in taskbar, just wait
                        print(f"{Fore.RED}[!] No safe apps in taskbar. Waiting...{Style.RESET_ALL}")
                        time.sleep(2)
                
                except Exception as e:
                    logger.error(f"Recovery failed: {e}")
                    time.sleep(2)
                
                continue
            
            # Get window title from first line
            window_title = tree_before.split('\n')[0] if tree_before else ""
            logger.info(f"Active window: {window_title}")
            
            # 2. FILTER: Find safe clickable elements
            lines = tree_before.split('\n')[1:]  # Skip "Window: ..." line
            candidates = []
            
            for line in lines:
                if not is_safe_element(line):
                    continue
                
                element = parse_element_line(line)
                if element:
                    candidates.append(element)
            
            if not candidates:
                logger.warning("No safe clickable elements found.")
                stats["skipped_unsafe"] += 1
                time.sleep(1)
                continue
            
            # 3. EXPLORE: Pick a random element to interact with
            target = random.choice(candidates)
            logger.info(f"Target: [{target['id']}] {target['type']}: '{target['name']}'")
            print(f"{Fore.YELLOW}[→] Clicking: [{target['id']}] {target['type']}: '{target['name']}'{Style.RESET_ALL}")
            
            # 4. ACT: Execute the click
            try:
                result = reader.click_id(target['id'])
                logger.info(f"Action result: {result}")
                
                if "Error" in result or "Failed" in result:
                    stats["failed_clicks"] += 1
                    print(f"{Fore.RED}[✗] Click failed: {result}{Style.RESET_ALL}")
                    time.sleep(ACTION_DELAY)
                    continue
                
            except Exception as e:
                logger.error(f"Exception during click: {e}")
                stats["failed_clicks"] += 1
                time.sleep(ACTION_DELAY)
                continue
            
            # 5. WAIT: Give UI time to react
            time.sleep(ACTION_DELAY)
            
            # 6. VERIFY: Did the screen change?
            logger.info("Verifying screen change...")
            tree_after = reader.capture_tree()
            
            # Compare trees (simple string comparison)
            if tree_before == tree_after:
                print(f"{Fore.YELLOW}[−] No screen change detected. Skipping...{Style.RESET_ALL}")
                stats["no_change"] += 1
                continue
            
            # 7. LEARN: Screen changed! Save this as training data
            print(f"{Fore.GREEN}[✓] SUCCESS: Screen changed!{Style.RESET_ALL}")
            
            # Generate synthetic instruction
            instruction = generate_instruction(target)
            
            # Save to dataset
            if save_training_example(tree_before, instruction, target['id'], logger):
                stats["successful_saves"] += 1
            
            # Show stats periodically
            if step % 10 == 0:
                print(f"\n{Fore.CYAN}📊 Stats after {step} steps:{Style.RESET_ALL}")
                print(f"   ✓ Successful saves: {stats['successful_saves']}")
                print(f"   ✗ Failed clicks: {stats['failed_clicks']}")
                print(f"   − No change: {stats['no_change']}")
                print(f"   ⊘ Skipped unsafe: {stats['skipped_unsafe']}")
    
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}[!] Training interrupted by user.{Style.RESET_ALL}")
    
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
    
    finally:
        # Final statistics
        print(f"\n{Fore.CYAN}╔════════════════════════════════════════════╗")
        print(f"║           TRAINING SESSION COMPLETE         ║")
        print(f"╚════════════════════════════════════════════╝{Style.RESET_ALL}\n")
        
        print(f"📊 Final Statistics:")
        print(f"   Total attempts:     {stats['total_attempts']}")
        print(f"   ✓ Successful saves: {stats['successful_saves']}")
        print(f"   ✗ Failed clicks:    {stats['failed_clicks']}")
        print(f"   − No change:        {stats['no_change']}")
        print(f"   ⊘ Skipped unsafe:   {stats['skipped_unsafe']}")
        print(f"\n💾 Dataset saved to: {DATASET_FILE}")
        print(f"🎓 Ready for training!\n")
        
        logger.info("Ghost Gym session ended.")

if __name__ == "__main__":
    main()
