"""
Ghost Actions: The Hands
"""

import logging
import pyautogui
import webbrowser
import subprocess
import time
from typing import Dict, Any

# Fail-Safe
pyautogui.FAILSAFE = True

logger = logging.getLogger("Hands")

class ActionController:
    """
    Executes actions decided by the Brain.
    Now focused on ID-based interaction (Blind God Mode).
    """
    
    def __init__(self, ui_reader):
        self.ui = ui_reader
        self.screen_width, self.screen_height = pyautogui.size()
        
    def execute(self, action_data: dict):
        """Execute an action, handling both 'action', 'tool' keys, and BATCH lists."""
        
        # --- DYNAMIC SAFETY SHIELD ---
        # Block dangerous actions based on keywords in args
        DANGEROUS_KEYWORDS = ["format", "delete", "wipe", "erase", "remove", "partition", "system32", "shutdown", "restart"]
        
        args = action_data.get("args", {})
        # Flatten args checking
        all_values = [str(v).lower() for v in args.values() if v]
        # Also check direct keys if flat structure
        all_values += [str(v).lower() for k,v in action_data.items() if k not in ["tool", "action"]]
        
        for val in all_values:
             if any(bad in val for bad in DANGEROUS_KEYWORDS):
                 logger.warning(f"🛡️ SAFETY SHIELD: BLOCKED attempt to '{val}'")
                 return f"Action Blocked: Safety keyword '{val}' detected."
        
        # --- CALM: BATCH EXECUTION SUPPORT ---
        if isinstance(action_data, list):
            logger.info(f"⚡ BATCH EXECUTION: Processing {len(action_data)} actions...")
            results = []
            for i, step in enumerate(action_data):
                result = self.execute(step)
                results.append(f"[{i+1}] {result}")
                time.sleep(0.5) # Short grace period between batch steps
            return " | ".join(results)
        
        # 1. Normalize the tool name
        # 1. Normalize the tool name
        act_type = action_data.get("tool") or action_data.get("action")
        args = action_data.get("args", {})
        
        # Handle flat structure (args at top level)
        if not args:
            args = {k: v for k, v in action_data.items() if k not in ["tool", "action"]}
        
        # 🛑 PATCH: Redirect "open" with an ID to "click"
        if act_type == "open" and "id" in args:
            act_type = "click"
            logger.info(f"PATCH: Redirected 'open' with ID to 'click'")
        
        # 🛑 PATCH: Redirect "open" with a name to "sys.launch"
        if act_type == "open" and ("app" in args or "name" in args):
            act_type = "sys.launch"
            logger.info(f"PATCH: Redirected 'open' with app to 'sys.launch'")
        
        logger.info(f"Executing: {act_type} -> {action_data}")
        
        if not act_type:
            return "Error: No action/tool specified."
                
        # --- TOOL: CLICK (ID-Based) ---
        if act_type in ["ui.click", "click"]:
            eid = args.get("id")
            if not eid:
                return "Error: Missing ID for click"
            
            # Validate ID is numeric
            try:
                eid = int(eid)
            except (ValueError, TypeError):
                return f"Error: ID must be a number, got '{eid}'. Use sys.launch for app names."
            
            return self.ui.click_id(eid)
                
        # --- TOOL: TYPE ---
        elif act_type in ["ui.type", "type"]:
            eid = args.get("id")
            text = args.get("text")
            if eid and text:
                return self.ui.type_in_id(int(eid), text)
            elif text:
                 # Fallback to blind typing if no ID
                 pyautogui.write(text)
                 return f"Typed '{text}' (Blind)"
            return "Error: No text provided."
        
        # --- TOOL: PRESS (Enter, Esc, etc) ---
        elif act_type == "key.press":
            key = args.get("key")
            if key:
                pyautogui.press(key)
                return f"Pressed {key}"
            return "Error: No key provided."
        
        # --- TOOL: SCROLL ---
        elif act_type in ["scroll", "scroll.down", "scroll.up"]:
            direction = args.get("direction", "down")  # default to down
            amount = args.get("amount", 3)  # number of scroll clicks
            
            # Handle action name variants
            if act_type == "scroll.down":
                direction = "down"
            elif act_type == "scroll.up":
                direction = "up"
            
            try:
                if direction == "down":
                    pyautogui.scroll(-amount * 120)  # negative = down
                    return f"Scrolled down {amount} clicks"
                elif direction == "up":
                    pyautogui.scroll(amount * 120)  # positive = up
                    return f"Scrolled up {amount} clicks"
                else:
                    return f"Error: Invalid scroll direction '{direction}'"
            except Exception as e:
                return f"Scroll failed: {e}"
        
        # --- TOOL: LAUNCH APP (The Fix) ---
        elif act_type in ["sys.launch", "launch"]:
            app_name = args.get("app") or args.get("name")
            
            if not app_name:
                return "Error: No app name provided."
            
            # Common shortcuts
            apps = {
                "calculator": "calc.exe",
                "notepad": "notepad.exe",
                "paint": "mspaint.exe",
                "chrome": "chrome.exe",
                "edge": "msedge.exe",
                "explorer": "explorer.exe",
                "cmd": "cmd.exe",
                "calc": "calc.exe"
            }
            
            # Try to map friendly name to exe, or use raw name
            target = apps.get(str(app_name).lower(), app_name)
            
            try:
                subprocess.Popen(target, shell=True)
                time.sleep(1)  # Give app time to launch
                return f"Launched {target}"
            except Exception as e:
                return f"Failed to launch: {e}"
            
        # --- TOOL: WAIT ---
        elif act_type == "wait":
            time.sleep(2)
            return "Waited 2s."
            
        # --- TOOL: DONE ---
        elif act_type == "done":
            return "DONE"
            
        # --- TOOL: BROWSER ---
        elif act_type == "browser.open":
            url = args.get("url")
            if url:
                webbrowser.open(url)
                return f"Opened {url}"
            return "Error: No URL provided."
            
        return f"Unknown action: {act_type}"
