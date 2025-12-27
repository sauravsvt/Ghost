"""
Ghost Web Controller - Playwright Browser Automation

THE RIGHT TOOL FOR WEB:
- Playwright reads the ACTUAL DOM, not pixels
- It can click by text: page.click("text=Search")
- It can type in fields: page.fill("input[name=search]", "query")
- It's FAST and ACCURATE

This is what we should have used from the start for web tasks.
"""

import logging
from typing import Optional, Dict, List, Any
from playwright.sync_api import sync_playwright, Page, Browser, Playwright

logger = logging.getLogger("WebController")


class WebController:
    """
    Browser automation using Playwright.
    
    This actually works for web pages because it reads the real DOM,
    not the Windows accessibility tree.
    """
    
    def __init__(self, headless: bool = False):
        """
        Initialize Playwright browser.
        
        Args:
            headless: Run browser without visible window
        """
        self.headless = headless
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None
        
    def start(self) -> str:
        """Start the browser."""
        try:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=self.headless)
            self._page = self._browser.new_page()
            logger.info("Playwright browser started")
            return "Browser started"
        except Exception as e:
            return f"Failed to start browser: {e}"
    
    def goto(self, url: str) -> str:
        """Navigate to a URL."""
        if not self._page:
            self.start()
        
        try:
            self._page.goto(url, wait_until="domcontentloaded", timeout=30000)
            return f"Navigated to {url}"
        except Exception as e:
            return f"Navigation failed: {e}"
    
    def get_elements(self, max_elements: int = 30) -> str:
        """
        Get interactive elements from the page as text.
        
        Returns something like:
            [0] Button: "Search"
            [1] Input: placeholder="Search..."
            [2] Link: "Hindi Songs"
        """
        if not self._page:
            return "[No page loaded]"
        
        try:
            # Get clickable elements
            elements = []
            self._element_locators = {}
            idx = 0
            
            # Buttons
            for btn in self._page.locator("button").all()[:10]:
                try:
                    text = btn.text_content().strip()[:50] if btn.text_content() else ""
                    if text:
                        elements.append(f"[{idx}] Button: \"{text}\"")
                        self._element_locators[idx] = btn
                        idx += 1
                except:
                    pass
            
            # Links
            for link in self._page.locator("a").all()[:10]:
                try:
                    text = link.text_content().strip()[:50] if link.text_content() else ""
                    if text and len(text) > 2:
                        elements.append(f"[{idx}] Link: \"{text}\"")
                        self._element_locators[idx] = link
                        idx += 1
                except:
                    pass
            
            # Input fields
            for inp in self._page.locator("input:visible").all()[:10]:
                try:
                    placeholder = inp.get_attribute("placeholder") or ""
                    name = inp.get_attribute("name") or inp.get_attribute("id") or ""
                    label = placeholder or name or "text field"
                    elements.append(f"[{idx}] Input: \"{label}\"")
                    self._element_locators[idx] = inp
                    idx += 1
                except:
                    pass
            
            # Video thumbnails (YouTube specific)
            for vid in self._page.locator("ytd-video-renderer, ytd-rich-item-renderer").all()[:5]:
                try:
                    title = vid.locator("#video-title").first.text_content()
                    if title:
                        title = title.strip()[:60]
                        elements.append(f"[{idx}] Video: \"{title}\"")
                        self._element_locators[idx] = vid.locator("#video-title").first
                        idx += 1
                except:
                    pass
            
            if not elements:
                return "[Page loaded but no interactive elements found]"
            
            url = self._page.url
            return f"[Page: {url}]\n" + "\n".join(elements[:max_elements])
            
        except Exception as e:
            return f"[Error reading page: {e}]"
    
    def click_id(self, element_id: int) -> str:
        """Click an element by its ID from the last get_elements call."""
        if element_id not in self._element_locators:
            return f"Error: ID {element_id} not found"
        
        try:
            locator = self._element_locators[element_id]
            locator.click()
            self._page.wait_for_timeout(500)  # Brief wait for page update
            return f"Clicked element {element_id}"
        except Exception as e:
            return f"Click failed: {e}"
    
    def type_text(self, element_id: int, text: str) -> str:
        """Type text into an element."""
        if element_id not in self._element_locators:
            return f"Error: ID {element_id} not found"
        
        try:
            locator = self._element_locators[element_id]
            locator.fill(text)
            return f"Typed '{text}'"
        except Exception as e:
            return f"Type failed: {e}"
    
    def press_key(self, key: str) -> str:
        """Press a key (e.g., 'Enter', 'Tab')."""
        try:
            self._page.keyboard.press(key)
            return f"Pressed {key}"
        except Exception as e:
            return f"Key press failed: {e}"
    
    def search_youtube(self, query: str) -> str:
        """Quick macro for YouTube search."""
        try:
            self.goto("https://www.youtube.com")
            self._page.wait_for_timeout(1000)
            
            # Find and click search box
            search = self._page.locator("input[name='search_query']")
            search.fill(query)
            search.press("Enter")
            
            # Wait for results
            self._page.wait_for_timeout(2000)
            
            return f"Searched YouTube for: {query}"
        except Exception as e:
            return f"YouTube search failed: {e}"
    
    def play_first_video(self) -> str:
        """Click the first video in YouTube results."""
        try:
            # Wait for video results
            self._page.wait_for_selector("ytd-video-renderer", timeout=5000)
            
            # Click first video
            first_video = self._page.locator("ytd-video-renderer #video-title").first
            first_video.click()
            
            self._page.wait_for_timeout(2000)
            return "Playing first video"
        except Exception as e:
            return f"Failed to play video: {e}"
    
    def screenshot(self, path: str = "screenshot.png") -> str:
        """Take a screenshot of the page."""
        try:
            self._page.screenshot(path=path)
            return f"Screenshot saved to {path}"
        except Exception as e:
            return f"Screenshot failed: {e}"
    
    def close(self):
        """Close the browser."""
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()
        logger.info("Browser closed")


# Quick test
if __name__ == "__main__":
    web = WebController(headless=False)
    print(web.start())
    print(web.search_youtube("hindi songs"))
    print(web.play_first_video())
    input("Press Enter to close...")
    web.close()
