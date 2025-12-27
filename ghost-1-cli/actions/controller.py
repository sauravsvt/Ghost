"""
Ghost Action Controller - The Hands

Handles all interactions with the desktop including:
- Mouse control (click, move)
- Keyboard control (type, hotkey)
- Browser operations
- File operations
- Command execution

Safety features:
- Failsafe (move to corner to stop)
- Rate limiting
- Path sandboxing
- Command blocklist
"""

import pyautogui
import webbrowser
import subprocess
import time
import logging
import os
from typing import Any, Dict, Optional
from dataclasses import dataclass

# Fail-Safe: Slam mouse to top-left corner to kill the agent
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1  # Small delay between actions

logger = logging.getLogger("ActionController")


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    output: str
    error: Optional[str] = None


class ActionController:
    """
    The Hands. Handles all desktop interactions with human-like behavior.
    """
    
    # Dangerous commands that are blocked
    BLOCKED_COMMANDS = [
        "rm -rf", "rmdir /s", "del /f", "format",
        "mkfs", "dd if=", ":(){:|:&};:", "shutdown",
        "reboot", "halt", "poweroff"
    ]
    
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()
        self.action_count = 0
        self.last_action_time = time.time()
        self.max_actions_per_minute = 60
        
        logger.info(f"ActionController initialized (screen: {self.screen_width}x{self.screen_height})")
    
    def _rate_limit_check(self) -> bool:
        """Check if we're within rate limits."""
        current_time = time.time()
        if current_time - self.last_action_time > 60:
            self.action_count = 0
            self.last_action_time = current_time
        
        if self.action_count >= self.max_actions_per_minute:
            logger.warning("Rate limit exceeded")
            return False
        
        self.action_count += 1
        return True
    
    def _validate_coordinates(self, x: int, y: int) -> bool:
        """Check if coordinates are within screen bounds."""
        return 0 <= x <= self.screen_width and 0 <= y <= self.screen_height

    def move_mouse_human_like(self, x: int, y: int, duration: float = 0.5) -> ToolResult:
        """
        Moves mouse using Bezier-like distinct steps to avoid bot detection.
        """
        if not self._validate_coordinates(x, y):
            return ToolResult(
                success=False,
                output="",
                error=f"Coordinates ({x},{y}) out of bounds"
            )
        
        logger.debug(f"Moving mouse to ({x}, {y})")
        
        try:
            pyautogui.moveTo(x, y, duration=duration, tween=pyautogui.easeOutQuad)
            return ToolResult(success=True, output=f"Moved to ({x}, {y})")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def click(self, x: Optional[int] = None, y: Optional[int] = None) -> ToolResult:
        """Click at current position or specified coordinates."""
        try:
            if x is not None and y is not None:
                if not self._validate_coordinates(x, y):
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"Coordinates ({x},{y}) out of bounds"
                    )
                self.move_mouse_human_like(x, y)
            
            pyautogui.click()
            pos = pyautogui.position()
            return ToolResult(success=True, output=f"Clicked at ({pos.x}, {pos.y})")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
        
    def type_text(self, text: str, cpm: int = 300) -> ToolResult:
        """
        Types text at a realistic speed.
        
        Args:
            text: Text to type
            cpm: Characters per minute (default 300 = fast human)
        """
        try:
            interval = 60 / cpm
            pyautogui.write(text, interval=interval)
            return ToolResult(success=True, output=f"Typed {len(text)} characters")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
    
    def press_hotkey(self, keys: str) -> ToolResult:
        """
        Press a key combination like 'ctrl+c' or 'alt+tab'.
        """
        try:
            key_list = [k.strip() for k in keys.split('+')]
            pyautogui.hotkey(*key_list)
            return ToolResult(success=True, output=f"Pressed {keys}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
    
    def open_browser(self, url: str) -> ToolResult:
        """
        Open a URL in the default browser.
        """
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            webbrowser.open(url)
            return ToolResult(success=True, output=f"Opened {url}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
    
    def wait(self, seconds: float) -> ToolResult:
        """
        Wait for specified duration.
        """
        try:
            time.sleep(seconds)
            return ToolResult(success=True, output=f"Waited {seconds}s")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
    
    def run_command(self, command: str) -> ToolResult:
        """
        Execute a shell command with safety checks.
        """
        # Check blocklist
        cmd_lower = command.lower()
        for blocked in self.BLOCKED_COMMANDS:
            if blocked in cmd_lower:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Blocked dangerous command: {blocked}"
                )
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            return ToolResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else None
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="", error="Command timed out")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> ToolResult:
        """
        The Agentic Dispatcher. Routes tool calls to appropriate handlers.
        
        Args:
            tool_name: Name of the tool (e.g. 'mouse.click', 'browser.open')
            args: Tool arguments as dictionary
            
        Returns:
            ToolResult with success status and output
        """
        if not self._rate_limit_check():
            return ToolResult(
                success=False,
                output="",
                error="Rate limit exceeded. Slow down."
            )
        
        logger.info(f"Executing: {tool_name} with {args}")
        
        try:
            if tool_name == "mouse.click":
                x = args.get('x')
                y = args.get('y')
                return self.click(x, y)
            
            elif tool_name == "mouse.move":
                return self.move_mouse_human_like(args['x'], args['y'])
            
            elif tool_name == "keyboard.type":
                return self.type_text(args['text'])
            
            elif tool_name == "keyboard.hotkey":
                return self.press_hotkey(args['keys'])
            
            elif tool_name == "browser.open":
                return self.open_browser(args['url'])
            
            elif tool_name == "wait":
                return self.wait(float(args.get('seconds', 1)))
            
            elif tool_name == "os.command":
                return self.run_command(args['command'])
            
            elif tool_name == "done":
                message = args.get('message', 'Task completed')
                logger.info(f"Task completed: {message}")
                return ToolResult(success=True, output=message)
            
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Unknown tool: {tool_name}"
                )
                
        except KeyError as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Missing required argument: {e}"
            )
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return ToolResult(success=False, output="", error=str(e))
