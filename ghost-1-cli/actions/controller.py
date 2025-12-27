"""
Ghost Action Controller - The Human Hands

SIMPLIFIED ARCHITECTURE (v3.0):
No more custom tool scripts. Just generic mouse/keyboard controls.
Gemini Vision predicts coordinates, we just execute.

This is how humans work:
1. Eyes see the button
2. Brain estimates location
3. Hands click it

We don't have a "click_youtube_play_button()" function in our brain.
"""

import pyautogui
import webbrowser
import subprocess
import time
import logging
from typing import Any, Dict, Optional
from dataclasses import dataclass

# Fail-Safe: Slam mouse to top-left corner to kill the agent
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1

logger = logging.getLogger("Hand")


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    output: str
    error: Optional[str] = None


class ActionController:
    """
    The Human Hands - Hybrid mouse/keyboard + UI Automation.
    
    v5.0 "Blind God" Mode:
    - ui.scan: Read UI tree (0.05s)
    - ui.click: Click by element ID (100% accurate)
    - ui.type: Type into element by ID
    
    Legacy Mode:
    - mouse.click: Pixel coordinates (fallback)
    """
    
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Initialize UI Automation reader
        try:
            from vision.structure import UIReader
            self.ui_reader = UIReader()
            logger.info("UIReader initialized (Blind God mode enabled)")
        except ImportError:
            self.ui_reader = None
            logger.warning("UIReader not available - falling back to mouse mode")
        
        logger.info(f"ActionController initialized (screen: {self.screen_width}x{self.screen_height})")
    
    def execute_tool(self, tool_name: str, args: Dict[str, Any], 
                     original_prompt: str = "") -> ToolResult:
        """
        Execute a tool based on AI-predicted actions.
        
        The AI sends ABSOLUTE pixel coordinates (1920x1080 screen).
        """
        logger.info(f"Executing: {tool_name} with {args}")
        
        try:
            # === UI AUTOMATION: Fast ID-based (v5.0 Blind God Mode) ===
            if tool_name == "ui.scan":
                if not self.ui_reader:
                    return ToolResult(success=False, output="", error="UIReader not available")
                tree = self.ui_reader.capture_tree()
                return ToolResult(success=True, output=tree)
            
            elif tool_name == "ui.click":
                if not self.ui_reader:
                    return ToolResult(success=False, output="", error="UIReader not available")
                idx = args.get("id")
                if idx is None:
                    return ToolResult(success=False, output="", error="Missing 'id' argument")
                result = self.ui_reader.click_id(int(idx))
                return ToolResult(success=True, output=result)
            
            elif tool_name == "ui.type":
                if not self.ui_reader:
                    return ToolResult(success=False, output="", error="UIReader not available")
                idx = args.get("id")
                text = args.get("text", "")
                if idx is None:
                    return ToolResult(success=False, output="", error="Missing 'id' argument")
                result = self.ui_reader.type_in_id(int(idx), text)
                return ToolResult(success=True, output=result)
            
            # === MOUSE: Direct pixel coordinates (legacy fallback) ===
            elif tool_name == "mouse.click":
                x = int(args.get("x", 0))
                y = int(args.get("y", 0))
                
                # Clamp to screen bounds
                x = max(0, min(x, self.screen_width))
                y = max(0, min(y, self.screen_height))
                
                # Move like a human (smooth motion)
                pyautogui.moveTo(x, y, duration=0.3)
                time.sleep(0.1)
                pyautogui.click()
                
                return ToolResult(success=True, output=f"Clicked at ({x}, {y})")
                    
            
            elif tool_name == "mouse.move":
                x = max(0, min(int(args.get("x", 0)), self.screen_width))
                y = max(0, min(int(args.get("y", 0)), self.screen_height))
                pyautogui.moveTo(x, y, duration=0.3)
                return ToolResult(success=True, output=f"Moved to ({x}, {y})")
            
            elif tool_name == "mouse.double_click":
                x = max(0, min(int(args.get("x", 0)), self.screen_width))
                y = max(0, min(int(args.get("y", 0)), self.screen_height))
                pyautogui.moveTo(x, y, duration=0.3)
                pyautogui.doubleClick()
                return ToolResult(success=True, output=f"Double-clicked at ({x}, {y})")
            
            elif tool_name == "mouse.scroll":
                direction = args.get("direction", "down")
                amount = args.get("amount", 3)
                clicks = amount if direction == "down" else -amount
                pyautogui.scroll(clicks)
                return ToolResult(success=True, output=f"Scrolled {direction} by {amount}")
            
            # === KEYBOARD: Direct input ===
            elif tool_name == "keyboard.type":
                text = args.get("text", "")
                pyautogui.write(text, interval=0.03)
                return ToolResult(success=True, output=f"Typed: {text}")
            
            elif tool_name == "keyboard.type_and_enter":
                text = args.get("text", "")
                pyautogui.write(text, interval=0.03)
                pyautogui.press("enter")
                return ToolResult(success=True, output=f"Typed and entered: {text}")
            
            elif tool_name == "keyboard.hotkey":
                keys = args.get("keys", "")
                key_list = [k.strip() for k in keys.split("+")]
                pyautogui.hotkey(*key_list)
                return ToolResult(success=True, output=f"Pressed: {keys}")
            
            elif tool_name == "keyboard.press":
                key = args.get("key", "enter")
                pyautogui.press(key)
                return ToolResult(success=True, output=f"Pressed: {key}")
            
            # === BROWSER: Just for starting sessions ===
            elif tool_name == "browser.open":
                url = args.get("url", "")
                if not url.startswith(("http://", "https://")):
                    url = "https://" + url
                webbrowser.open(url)
                return ToolResult(success=True, output=f"Opened browser: {url}")
            
            # === WAIT: For page loads ===
            elif tool_name == "wait":
                seconds = float(args.get("seconds", 1))
                time.sleep(seconds)
                return ToolResult(success=True, output=f"Waited {seconds}s")
            
            # === CODE EXECUTION: For complex tasks ===
            elif tool_name == "python.exec":
                code = args.get("code", "")
                if not code:
                    return ToolResult(success=False, output="", error="No code provided")
                
                # Security: Block dangerous operations
                dangerous = ["os.system", "subprocess", "rmtree", "remove(", "unlink("]
                for pattern in dangerous:
                    if pattern in code:
                        return ToolResult(success=False, output="", 
                                         error=f"Blocked: {pattern}")
                
                try:
                    import io
                    import sys
                    old_stdout = sys.stdout
                    sys.stdout = io.StringIO()
                    exec(code, {"__builtins__": __builtins__})
                    output = sys.stdout.getvalue()
                    sys.stdout = old_stdout
                    return ToolResult(success=True, output=output or "Executed")
                except Exception as e:
                    return ToolResult(success=False, output="", error=str(e))
            
            # === DONE: Task completion ===
            elif tool_name == "done":
                message = args.get("message", "Task completed")
                return ToolResult(success=True, output=message)
            
            else:
                return ToolResult(success=False, output="", error=f"Unknown tool: {tool_name}")
                
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return ToolResult(success=False, output="", error=str(e))
    
    # Backwards compatibility alias
    def execute(self, action_data: Dict[str, Any]) -> str:
        """Legacy execute method for simpler interface."""
        tool = action_data.get("tool", "")
        args = action_data.get("args", {})
        result = self.execute_tool(tool, args)
        return result.output if result.success else f"Error: {result.error}"
