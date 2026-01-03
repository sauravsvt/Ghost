"""
Ghost Actions: The Hands
"""

import logging
import pyautogui
import webbrowser
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
        
    def execute(self, action: Dict[str, Any]):
        """
        Execute JSON action from Brain.
        """
        act_type = action.get("action")
        
        logger.info(f"Executing: {act_type} -> {action}")
        
        if act_type == "click":
            # [ID] Click
            eid = action.get("id")
            if eid:
                return self.ui.click_id(int(eid))
            else:
                return "Error: No ID provided for click."
                
        elif act_type == "type":
            # [ID] Type
            eid = action.get("id")
            text = action.get("text")
            if eid and text:
                return self.ui.type_in_id(int(eid), text)
            elif text:
                 # Fallback to blind typing if no ID
                 pyautogui.write(text)
                 return f"Typed '{text}' (Blind)"
            return "Error: No text provided."
            
        elif act_type == "wait":
            time.sleep(2)
            return "Waited 2s."
            
        elif act_type == "done":
            return "DONE"
            
        elif act_type == "browser.open":
            # Helper for starting
            url = action.get("url")
            webbrowser.open(url)
            return f"Opened {url}"
            
        return f"Unknown action: {act_type}"
