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
        Scans the active window and returns a numbered list of interactable elements.
        Format:
        [1] Button: 'Search'
        [2] Edit: 'Type here...'
        """
        try:
            # 1. Get Active Window
            desktop = Desktop(backend="uia")
            window = desktop.window(active_only=True)
            
            if not window.exists():
                return "No active window found."
            
            window_title = window.window_text()
            
            # 2. Walk the Tree
            elements = []
            self.element_map.clear()
            
            # Filter for interactable types
            interesting_types = [
                "Button", "Edit", "ListItem", "MenuItem", 
                "TabItem", "CheckBox", "RadioButton", 
                "ComboBox", "Hyperlink", "Document"
            ]
            
            # Get all descendants (flattened)
            # We filter by control_type and visibility
            descendants = window.descendants()
            
            current_id = 1
            tree_lines = [f"Window: {window_title}"]
            
            for child in descendants:
                if not child.is_visible():
                    continue
                    
                c_type = child.friendly_class_name()
                c_title = child.window_text()
                
                # Filter noise
                if c_type not in interesting_types:
                    continue
                    
                # Store mapped element
                self.element_map[current_id] = child
                
                # Format: [ID] Type: 'Name'
                line = f"[{current_id}] {c_type}: '{c_title}'"
                tree_lines.append(line)
                
                current_id += 1
                
                # Safety cap to prevent token explosion
                if current_id > 100:
                    tree_lines.append("... (more elements truncated)")
                    break
            
            self.last_tree = "\n".join(tree_lines)
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
