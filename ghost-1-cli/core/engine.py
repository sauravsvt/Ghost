"""
Ghost Engine v3.0 - Gemini Cloud Brain

This replaces the local Qwen + Moondream stack with a single
Gemini API call that handles both vision AND reasoning.

Trade-off: Privacy for Supercomputer Capabilities
- 8K resolution understanding
- Multi-step planning
- Code generation
"""

import os
import base64
import json
import re
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load env variables (API Key)
load_dotenv()

logger = logging.getLogger("GhostEngine")

# System prompt - Simple natural protocol with ABSOLUTE coordinates
SYSTEM_PROMPT = """You are Ghost-1, an autonomous computer agent.
You see the screen and control the mouse/keyboard.

PROTOCOL:
1. If you need to ACT (click, type, scroll, open), output JSON ONLY.
2. If you are DONE or need to talk to the user, output PLAIN TEXT only.
3. Do NOT stop until the user's goal is FULLY complete.

COORDINATE SYSTEM (IMPORTANT):
- Screen resolution is 1920x1080 pixels
- Use ABSOLUTE pixel coordinates (not normalized)
- (0,0) = top-left corner, (1920,1080) = bottom-right corner

YOUTUBE LAYOUT GUIDE (1920x1080):
- Search bar: x=600-700, y=40-50 (top center, very close to top edge)
- First video thumbnail in search results: x=400-500, y=300-400
- Video player: center of screen when playing

COMMON UI PATTERNS:
- Browser URL bar: x=400-600, y=50-80
- Windows Start button: x=30, y=1050 (bottom-left)
- Close button (X): top-right corner of windows

TOOLS:
{"tool": "browser.open", "args": {"url": "https://..."}, "reasoning": "..."}
{"tool": "mouse.click", "args": {"x": 650, "y": 45}, "reasoning": "Clicking YouTube search bar"}
{"tool": "keyboard.type_and_enter", "args": {"text": "..."}, "reasoning": "..."}
{"tool": "keyboard.type", "args": {"text": "..."}, "reasoning": "..."}
{"tool": "keyboard.press", "args": {"key": "enter"}, "reasoning": "..."}
{"tool": "mouse.scroll", "args": {"direction": "down", "amount": 3}, "reasoning": "..."}
{"tool": "wait", "args": {"seconds": 2}, "reasoning": "..."}

EXAMPLE - Playing song on YouTube:
Step 1: {"tool": "browser.open", "args": {"url": "https://youtube.com"}, "reasoning": "Opening YouTube"}
Step 2: {"tool": "mouse.click", "args": {"x": 650, "y": 45}, "reasoning": "Clicking search bar at top"}
Step 3: {"tool": "keyboard.type_and_enter", "args": {"text": "hindi songs"}, "reasoning": "Searching"}
Step 4: {"tool": "mouse.click", "args": {"x": 450, "y": 350}, "reasoning": "Clicking first video"}
Step 5: "The song is now playing on YouTube."

Remember: JSON = keep working. Plain text = you're done."""


class GhostEngine:
    """
    The Cloud Brain - Gemini API for Vision + Reasoning.
    
    Replaces both Moondream (vision) and Qwen (reasoning) with a single
    Gemini API call that can understand images and plan actions.
    """
    
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("❌ GEMINI_API_KEY missing! Add it to your .env file.")
        
        # Import here to avoid loading if not used
        try:
            from google import genai
            from google.genai import types
            self.genai = genai
            self.types = types
        except ImportError:
            raise ImportError("❌ google-genai not installed. Run: pip install google-genai")
        
        logger.info("Connecting to Gemini Cloud Brain...")
        print("[*] Connecting to Gemini Cloud Brain...")
        
        self.client = genai.Client(api_key=self.api_key)
        
        # Model selection - try experimental thinking model first
        # Fallback chain: thinking-exp -> flash-exp -> flash
        self.model = "gemini-3-pro-preview"
        self.fallback_models = [
            "gemini-flash-latest",
            "gemini-flash-lite-latest"
        ]
        
        print(f"[*] Using model: {self.model}")
        logger.info(f"Gemini client initialized with model: {self.model}")
    
    def think(self, user_prompt: str, screen_image_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Send text + screen (if available) to Gemini.
        
        Args:
            user_prompt: What the user wants to do
            screen_image_bytes: JPEG bytes of the screen (optional)
            
        Returns:
            Dict with 'action' (parsed JSON) and 'raw_response'
        """
        types = self.types
        parts = []
        
        # 1. Add Image (if we have one)
        if screen_image_bytes:
            parts.append(types.Part.from_bytes(
                data=screen_image_bytes, 
                mime_type="image/jpeg"
            ))
            logger.info("Added screen image to request")
        
        # 2. Add the prompt (system + user combined)
        full_prompt = f"{SYSTEM_PROMPT}\n\n---\nUser Request: {user_prompt}\n---\n\nRespond with JSON only:"
        parts.append(types.Part.from_text(text=full_prompt))
        
        # 3. Call API with retry logic
        response_text = self._call_api(parts)
        
        # 4. Parse JSON from response
        action = self._parse_json(response_text)
        
        return {
            "action": action,
            "raw_response": response_text,
            "reasoning": action.get("reasoning", "") if action else ""
        }
    
    def think_fast(self, user_prompt: str, screen_context: str = "") -> Dict[str, Any]:
        """
        Fast thinking without image upload.
        Used for simple commands where vision isn't needed.
        """
        return self.think(user_prompt, screen_image_bytes=None)
    
    def think_text(self, context: str) -> str:
        """
        Pure text thinking - for UI tree based reasoning.
        
        This is the FAST path:
        - No image upload
        - Just text in, text out
        - Perfect for UI element selection
        
        Args:
            context: The full context including UI tree and task
            
        Returns:
            Raw response text (JSON or plain text)
        """
        types = self.types
        
        # Simple text prompt - no vision needed
        prompt = f"""You are Ghost-1, a UI automation agent.
You will be given a list of UI elements and a task.
Pick the right element ID to click, or speak if done.

RULES:
- To click an element: {{"tool": "ui.click", "id": NUMBER}}
- To type in a field: {{"tool": "ui.type", "id": NUMBER, "text": "..."}}
- To open a URL: {{"tool": "browser.open", "url": "https://..."}}
- To type anywhere: {{"tool": "keyboard.type", "text": "..."}}
- To press a key: {{"tool": "keyboard.press", "key": "enter"}}
- If DONE: Just say it in plain text (no JSON)

Output JSON to continue working, or plain text when finished.

{context}"""
        
        parts = [types.Part.from_text(text=prompt)]
        response = self._call_api(parts)
        
        return response or ""
    
    def see(self, image_bytes: bytes, question: str = "What do you see?") -> str:
        """
        Use Gemini's vision to describe the screen.
        """
        types = self.types
        parts = [
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            types.Part.from_text(text=f"Describe this screen briefly: {question}")
        ]
        
        return self._call_api(parts)
    
    def _call_api(self, parts: list) -> str:
        """Call Gemini API with fallback to other models if needed."""
        types = self.types
        
        models_to_try = [self.model] + self.fallback_models
        last_error = None
        
        for model in models_to_try:
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=[types.Content(role="user", parts=parts)],
                    config=types.GenerateContentConfig(
                        temperature=0.1,  # Precise actions
                        max_output_tokens=1024
                    )
                )
                
                if model != self.model:
                    logger.info(f"Using fallback model: {model}")
                    self.model = model  # Update for future calls
                
                return response.text
                
            except Exception as e:
                last_error = e
                logger.warning(f"Model {model} failed: {e}")
                continue
        
        return f"API Error: {last_error}"
    
    def _parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from response text (handles markdown code blocks)."""
        if not text:
            return None
        
        # Try to find JSON in code blocks first
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find raw JSON
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        return None
    
    def log_success(self, task: str, result: Dict[str, Any]):
        """Log successful interactions (placeholder for learning)."""
        logger.info(f"Success: {task}")


# For backwards compatibility
class ModelConfig:
    """Dummy config for backwards compatibility."""
    pass
