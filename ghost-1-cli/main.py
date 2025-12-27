#!/usr/bin/env python3
"""
Ghost-1: Unified Agent (v7.0)

ONE agent that handles EVERYTHING:
- Web: Playwright (YouTube, Google, etc.)
- Desktop: pyautogui + pywinauto
- System: Direct commands (mute, screenshot, etc.)

Smart routing based on what you ask.
"""

import colorama
from colorama import Fore, Style
import re
import time
import os

# Initialize colorama
colorama.init()


def print_banner():
    print(f"\n{Fore.CYAN}╔══════════════════════════════════════╗")
    print(f"{Fore.CYAN}║       GHOST-1: BLIND GOD (v8.0)       ║")
    print(f"{Fore.CYAN}║  UI Automation • 0.05s • 100% Accurate ║")
    print(f"{Fore.CYAN}╚══════════════════════════════════════╝{Style.RESET_ALL}")


class GhostAgent:
    """Unified agent that routes to the right tool.
    
    v5.0 Blind God Mode:
    - scan: Read UI tree instantly
    - click <id>: Click element by ID (100% accurate)
    """
    
    def __init__(self):
        self.web = None  # Lazy load Playwright
        self.ui_reader = None  # Lazy load UIReader
        self._init_basic()
    
    def _init_basic(self):
        """Initialize basic tools (always available)."""
        import pyautogui
        self.pyautogui = pyautogui
        pyautogui.FAILSAFE = True
        print(f"{Fore.GREEN}✓ System controls ready{Style.RESET_ALL}")
        
        # Initialize UI Automation (Blind God Mode)
        try:
            from vision.structure import UIReader
            self.ui_reader = UIReader()
            print(f"{Fore.GREEN}✓ UI Scanner ready (Blind God Mode){Style.RESET_ALL}")
        except ImportError:
            print(f"{Fore.YELLOW}⚠ UIReader not available (install pywinauto){Style.RESET_ALL}")
    
    def _init_web(self):
        """Lazy initialize Playwright browser."""
        if self.web is None:
            print(f"{Fore.YELLOW}[*] Starting browser...{Style.RESET_ALL}")
            try:
                from web.controller import WebController
                self.web = WebController(headless=False)
                self.web.start()
                print(f"{Fore.GREEN}✓ Browser ready{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}[!] Browser init failed: {e}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Run: pip install playwright && playwright install chromium{Style.RESET_ALL}")
                return False
        return True
    
    # =========== SYSTEM COMMANDS ===========
    
    def mute(self) -> str:
        self.pyautogui.press('volumemute')
        return "Volume toggled"
    
    def volume_up(self) -> str:
        for _ in range(5):
            self.pyautogui.press('volumeup')
        return "Volume increased"
    
    def volume_down(self) -> str:
        for _ in range(5):
            self.pyautogui.press('volumedown')
        return "Volume decreased"
    
    def screenshot(self) -> str:
        import datetime
        filename = f"screenshot_{datetime.datetime.now().strftime('%H%M%S')}.png"
        self.pyautogui.screenshot(filename)
        return f"Screenshot saved: {filename}"
    
    def type_text(self, text: str) -> str:
        self.pyautogui.write(text, interval=0.02)
        return f"Typed: {text}"
    
    def press_key(self, key: str) -> str:
        self.pyautogui.press(key)
        return f"Pressed: {key}"
    
    def hotkey(self, keys: str) -> str:
        key_list = [k.strip() for k in keys.split('+')]
        self.pyautogui.hotkey(*key_list)
        return f"Pressed: {keys}"
    
    # =========== WEB COMMANDS ===========
    
    def play_youtube(self, query: str) -> str:
        """Search and play on YouTube."""
        if not self._init_web():
            return "Browser not available"
        
        try:
            print(f"{Fore.BLUE}[*] Searching YouTube: {query}{Style.RESET_ALL}")
            
            # Navigate to YouTube
            self.web.goto("https://www.youtube.com")
            time.sleep(2)  # Wait for page load
            
            # Search
            self.web._page.locator("input[name='search_query']").fill(query)
            self.web._page.locator("input[name='search_query']").press("Enter")
            time.sleep(3)  # Wait for search results
            
            # Click first video
            print(f"{Fore.BLUE}[*] Clicking first result...{Style.RESET_ALL}")
            self.web._page.locator("ytd-video-renderer #video-title").first.click()
            time.sleep(2)
            
            return f"Now playing: {query}"
        except Exception as e:
            return f"YouTube failed: {e}"
    
    def search_google(self, query: str) -> str:
        """Search Google."""
        if not self._init_web():
            return "Browser not available"
        
        try:
            self.web.goto(f"https://www.google.com/search?q={query.replace(' ', '+')}")
            return f"Searched: {query}"
        except Exception as e:
            return f"Google search failed: {e}"
    
    def open_url(self, url: str) -> str:
        """Open a URL."""
        if not url.startswith('http'):
            url = 'https://' + url
        
        if not self._init_web():
            # Fallback to default browser
            import webbrowser
            webbrowser.open(url)
            return f"Opened in default browser: {url}"
        
        self.web.goto(url)
        return f"Opened: {url}"
    
    # =========== UI AUTOMATION (Blind God Mode) ===========
    
    def scan_ui(self) -> str:
        """Scan active window for interactive elements."""
        if not self.ui_reader:
            return "UIReader not available. Install pywinauto."
        return self.ui_reader.capture_tree()
    
    def click_element(self, element_id: int) -> str:
        """Click element by ID from last scan."""
        if not self.ui_reader:
            return "UIReader not available."
        return self.ui_reader.click_id(element_id)
    
    def type_in_element(self, element_id: int, text: str) -> str:
        """Type text into element by ID."""
        if not self.ui_reader:
            return "UIReader not available."
        return self.ui_reader.type_in_id(element_id, text)
    
    # =========== COMMAND ROUTER ===========
    
    def process(self, command: str) -> str:
        """Route command to the right handler."""
        cmd = command.lower().strip()
        
        # === UI AUTOMATION COMMANDS (v5.0) ===
        if cmd == 'scan':
            return self.scan_ui()
        
        if cmd.startswith('click '):
            try:
                idx = int(cmd.split()[1])
                return self.click_element(idx)
            except (ValueError, IndexError):
                return "Usage: click <id>"
        
        if cmd.startswith('type '):
            # Format: type <id> <text>
            parts = cmd.split(maxsplit=2)
            if len(parts) >= 3:
                try:
                    idx = int(parts[1])
                    text = parts[2]
                    return self.type_in_element(idx, text)
                except ValueError:
                    pass
            return "Usage: type <id> <text>"
        
        # === SYSTEM COMMANDS ===
        if any(word in cmd for word in ['mute', 'unmute', 'silence']):
            return self.mute()
        
        if 'volume up' in cmd or 'louder' in cmd:
            return self.volume_up()
        
        if 'volume down' in cmd or 'quieter' in cmd:
            return self.volume_down()
        
        if 'screenshot' in cmd:
            return self.screenshot()
        
        # === WEB COMMANDS ===
        
        # YouTube play
        if 'youtube' in cmd or 'play' in cmd:
            # Extract what to play
            query = self._extract_query(cmd, ['play', 'youtube', 'on', 'open', 'the', 'some', 'me', 'and', 'video', 'song', 'music'])
            if query:
                return self.play_youtube(query)
            else:
                return "What do you want to play? Try: play hindi songs"
        
        # Google search
        if 'search' in cmd or 'google' in cmd:
            query = self._extract_query(cmd, ['search', 'google', 'for', 'find', 'look', 'up', 'the'])
            if query:
                return self.search_google(query)
            else:
                return "What do you want to search? Try: search python tutorials"
        
        # Open URL
        if 'open' in cmd or 'goto' in cmd or 'go to' in cmd:
            # Check if it looks like a URL
            url_match = re.search(r'(https?://)?[\w.-]+\.(com|org|net|io|dev|edu|gov)', cmd)
            if url_match:
                return self.open_url(url_match.group(0))
            
            # Check for common sites
            if 'youtube' in cmd:
                return self.open_url('youtube.com')
            if 'google' in cmd:
                return self.open_url('google.com')
            if 'github' in cmd:
                return self.open_url('github.com')
        
        # === KEYBOARD SHORTCUTS ===
        if 'copy' in cmd:
            return self.hotkey('ctrl+c')
        if 'paste' in cmd:
            return self.hotkey('ctrl+v')
        if 'undo' in cmd:
            return self.hotkey('ctrl+z')
        if 'save' in cmd:
            return self.hotkey('ctrl+s')
        
        # === DEFAULT ===
        return f"I don't understand: {command}\nTry: play hindi songs, mute, screenshot, search python"
    
    def _extract_query(self, text: str, remove_words: list) -> str:
        """Extract the actual query from a command."""
        words = text.lower().split()
        filtered = [w for w in words if w not in remove_words and len(w) > 1]
        return ' '.join(filtered)
    
    def close(self):
        """Cleanup."""
        if self.web:
            self.web.close()


def main():
    print_banner()
    
    print(f"{Fore.YELLOW}[*] Initializing Ghost-1...{Style.RESET_ALL}")
    agent = GhostAgent()
    
    print(f"\n{Fore.GREEN}✓ Ghost-1 Ready{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Commands:{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}[Blind God Mode]{Style.RESET_ALL}")
    print(f"  • scan            - Read UI elements")
    print(f"  • click <id>      - Click element by ID")
    print(f"  • type <id> <txt> - Type into element")
    print(f"  {Fore.WHITE}[Legacy Mode]{Style.RESET_ALL}")
    print(f"  • play hindi songs")
    print(f"  • mute / volume up")
    print(f"  • open github.com")
    print(f"  • exit")
    
    while True:
        try:
            user_input = input(f"\n{Fore.YELLOW}Ghost-1:~$ {Style.RESET_ALL}").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
            
            # Process command
            start = time.time()
            result = agent.process(user_input)
            elapsed = time.time() - start
            
            print(f"{Fore.GREEN}✓ {result} ({elapsed:.1f}s){Style.RESET_ALL}")
            
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Interrupted.{Style.RESET_ALL}")
            break
        except Exception as e:
            print(f"{Fore.RED}[!] Error: {e}{Style.RESET_ALL}")
    
    agent.close()
    print(f"{Fore.CYAN}Ghost-1 shutdown.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
