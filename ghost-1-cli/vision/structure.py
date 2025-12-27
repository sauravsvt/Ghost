"""
Ghost Vision v5.0 - Structure Reader (UI Accessibility)

THE PIVOT:
- Vision = Slow (5-10 seconds), Inaccurate (pixel guessing)
- Structure = Instant (<0.1 seconds), 100% Accurate (click by ID)

This reads the Windows UI Accessibility tree, which contains:
- Buttons, Links, TextFields, etc.
- Their names, types, and positions
- All as TEXT - perfect for LLMs

The LLM sees:
  [0] Button: 'Search'
  [1] Edit: 'Search Query'  
  [2] Link: 'Hindi Songs - YouTube'

And outputs:
  {"tool": "ui.click", "id": 2}

INSTANT. ACCURATE. TRAINABLE.
"""

import logging
from typing import Dict, List, Optional, Any
import time

logger = logging.getLogger("UIReader")


class UIReader:
    """
    Reads the UI Accessibility tree and returns a text representation.
    
    This is 100x faster than vision because:
    1. No screenshot capture
    2. No API upload
    3. No coordinate guessing
    """
    
    # Control types we care about (interactive elements)
    INTERACTIVE_TYPES = [
        "Button", "Edit", "ComboBox", "CheckBox", 
        "RadioButton", "Link", "ListItem", "MenuItem",
        "TabItem", "Text", "Hyperlink"
    ]
    
    def __init__(self):
        self.element_map: Dict[int, Any] = {}
        self._backend = "uia"  # Windows UI Automation
        
        try:
            from pywinauto import Desktop
            self.Desktop = Desktop
            logger.info("UIReader initialized (pywinauto UIA backend)")
        except ImportError:
            raise ImportError("pywinauto not installed. Run: pip install pywinauto")
    
    def capture_tree(self, max_elements: int = 50, depth: int = 5) -> str:
        """
        Capture the UI tree of the active window as text.
        
        Returns a string like:
            [0] Button: 'Back'
            [1] Edit: 'Search Box'
            [2] Button: 'Search'
            [3] Link: 'Hindi Songs'
        
        Args:
            max_elements: Maximum elements to return (for speed)
            depth: How deep to search in the UI tree
        """
        self.element_map.clear()
        
        try:
            start = time.time()
            
            # Get active window
            desktop = self.Desktop(backend=self._backend)
            active = desktop.window(active_only=True)
            
            if not active.exists():
                return "[No active window detected]"
            
            window_title = active.window_text()
            elements = []
            idx = 0
            
            # Collect interactive elements only
            for control_type in self.INTERACTIVE_TYPES:
                try:
                    descendants = active.descendants(control_type=control_type, depth=depth)
                    for item in descendants:
                        if idx >= max_elements:
                            break
                        
                        name = item.window_text().strip()
                        if name and len(name) > 1 and len(name) < 100:
                            # Format: [ID] Type: 'Name'
                            elements.append(f"[{idx}] {control_type}: '{name}'")
                            self.element_map[idx] = item
                            idx += 1
                except Exception:
                    continue
            
            elapsed = time.time() - start
            logger.info(f"UI tree captured: {idx} elements in {elapsed:.2f}s")
            
            if not elements:
                return f"[Window: {window_title}]\n(No interactive elements found)"
            
            header = f"[Window: {window_title}]\n"
            return header + "\n".join(elements)
            
        except Exception as e:
            logger.error(f"UI capture failed: {e}")
            return f"[Error reading UI: {e}]"
    
    def click_id(self, element_id: int) -> str:
        """Click an element by its ID from the last capture."""
        if element_id not in self.element_map:
            return f"Error: ID {element_id} not found. Run capture_tree first."
        
        try:
            element = self.element_map[element_id]
            element.click_input()
            name = element.window_text()
            return f"Clicked [{element_id}]: '{name}'"
        except Exception as e:
            return f"Click failed: {e}"
    
    def type_in_id(self, element_id: int, text: str) -> str:
        """Type text into an element by ID."""
        if element_id not in self.element_map:
            return f"Error: ID {element_id} not found."
        
        try:
            element = self.element_map[element_id]
            element.click_input()  # Focus first
            element.type_keys(text, with_spaces=True)
            return f"Typed '{text}' into [{element_id}]"
        except Exception as e:
            return f"Type failed: {e}"
    
    def get_element_info(self, element_id: int) -> Optional[Dict]:
        """Get detailed info about an element."""
        if element_id not in self.element_map:
            return None
        
        try:
            element = self.element_map[element_id]
            rect = element.rectangle()
            return {
                "id": element_id,
                "name": element.window_text(),
                "type": element.friendly_class_name(),
                "x": rect.left,
                "y": rect.top,
                "width": rect.width(),
                "height": rect.height()
            }
        except:
            return None


# Backwards compatibility alias
class Eye(UIReader):
    """Alias for backwards compatibility."""
    def capture_raw(self) -> bytes:
        """Legacy method - returns tree as bytes for compatibility."""
        return self.capture_tree().encode('utf-8')


class ScreenPerceptor(UIReader):
    """Alias for backwards compatibility."""
    pass
