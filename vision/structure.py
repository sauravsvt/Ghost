"""
Ghost-1 Vision: The Matrix Reader (Structure over Pixels)
"""

import logging
from typing import Dict, List, Optional, Any
from pywinauto import Desktop


logger = logging.getLogger("Vision")

class UIReader:
    """
    Scans the UI using Windows Accessibility API (UIA).
    Converts 'dumb' pixels into structured text (The Matrix).
    """
    
    def __init__(self):
        self.element_map: Dict[int, Any] = {}
        self.last_tree: str = ""
        
    def capture_tree(self) -> str:
        """
        Capture UI tree with aggressive filtering to save context.
        Max 60 elements, skip containers, truncate long names.
        """
        try:
            app = Desktop(backend="uia").window(active_only=True)
            if not app.exists():
                return "No active window."
            
            window_title = app.window_text() or "Unknown"
            elements = [f"Window: {window_title}"]
            self.element_map = {}
            idx = 1
            
            # Get all controls
            controls = app.descendants()
            
            for item in controls:
                # STOP if we have too many (Save Tokens!)
                if idx > 60:
                    break
                
                try:
                    name = item.window_text()
                    control_type = item.friendly_class_name()
                    
                    # Skip boring containers (50% token reduction!)
                    if control_type in ["Pane", "Group", "Window", "Image"]:
                        continue
                    
                    # Skip empty text unless it's an Edit box
                    if not name and control_type != "Edit":
                        continue
                    
                    # Truncate long names (e.g. browser tab titles)
                    if len(name) > 50:
                        name = name[:47] + "..."
                    
                    if item.is_visible():
                        label = f"[{idx}] {control_type}: '{name}'"
                        elements.append(label)
                        self.element_map[idx] = item
                        idx += 1
                        
                except Exception:
                    continue
            
            self.last_tree = "\n".join(elements)
            return self.last_tree
            
        except Exception as e:
            logger.error(f"Scan failed: {e}")
            return f"Error scanning UI: {e}"

    def click_id(self, element_id: int) -> str:
        """Click an element by its ID from the last scan."""
        if element_id not in self.element_map:
            return f"Error: ID [{element_id}] not found in last scan."
            
        try:
            element = self.element_map[element_id]
            name = element.window_text()
            
            # UIA Click
            try:
                element.click_input() # Usually more reliable for specialized UI
            except:
                 element.invoke() # Fallback
            
            return f"Clicked [{element_id}] '{name}'"
        except Exception as e:
            return f"Failed to click [{element_id}]: {e}"

    def type_in_id(self, element_id: int, text: str) -> str:
        """Type text into an element."""
        if element_id not in self.element_map:
            return f"Error: ID [{element_id}] not found."
            
        try:
            element = self.element_map[element_id]
            element.set_focus()
            element.type_keys(text, with_spaces=True)
            return f"Typed '{text}' into [{element_id}]"
        except Exception as e:
            return f"Failed to type: {e}"
    
    def execute_id(self, element_id: int) -> str:
        """Alias for click_id for backward compatibility."""
        return self.click_id(element_id)
