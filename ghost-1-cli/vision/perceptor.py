"""
Ghost Vision Engine - True Sight with Moondream2 VLM

This module provides real visual understanding of the screen using
Moondream2, a tiny but capable Vision-Language Model.

Features:
    - Screen capture to PIL Image conversion
    - Visual Question Answering (describe what's on screen)
    - Object/Element pointing (get coordinates of UI elements)
    - Grid-based coordinate mapping

Model: vikhyatk/moondream2 (revision 2024-08-26)
RAM: ~2GB
Speed: 2-4 seconds per query on CPU
"""

import mss
import numpy as np
import logging
from PIL import Image
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
import io

logger = logging.getLogger("ScreenPerceptor")


@dataclass
class VisualElement:
    """Represents a detected UI element with its location."""
    description: str
    x: int
    y: int
    confidence: float = 1.0


class ScreenPerceptor:
    """
    The Eyes. Captures screen and provides visual understanding via VLM.
    
    Uses Moondream2 for:
    - Screen description (what's visible)
    - Element location (where is the search box?)
    - UI understanding (is there an error message?)
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize screen capture and VLM.
        
        Args:
            use_gpu: Use CUDA if available (faster but requires GPU)
        """
        self.sct = mss.mss()
        self.vlm = None
        self.device = "cuda" if use_gpu else "cpu"
        self._screen_width = 0
        self._screen_height = 0
        
        # Load VLM
        self._load_vlm()
        
    def _load_vlm(self):
        """Load Moondream2 Vision-Language Model."""
        try:
            from transformers import AutoModelForCausalLM
            import torch
            
            logger.info("Loading Moondream2 VLM (first run downloads ~2GB)...")
            
            # Use CPU-friendly settings
            self.vlm = AutoModelForCausalLM.from_pretrained(
                "vikhyatk/moondream2",
                revision="2024-08-26",
                trust_remote_code=True,
                torch_dtype=torch.float32,  # CPU compatible
                device_map={"": self.device}
            )
            
            logger.info(f"Moondream2 loaded on {self.device}")
            
        except ImportError as e:
            logger.error(f"Failed to load VLM: {e}")
            logger.error("Install with: pip install transformers torch einops")
            self.vlm = None
        except Exception as e:
            logger.error(f"VLM loading error: {e}")
            self.vlm = None
    
    def capture_state(self) -> np.ndarray:
        """
        Capture the primary monitor screen.
        
        Returns:
            numpy array of shape (H, W, 3) in RGB format
        """
        monitor = self.sct.monitors[1]  # Primary monitor
        screenshot = self.sct.grab(monitor)
        
        # Update screen dimensions
        self._screen_width = screenshot.width
        self._screen_height = screenshot.height
        
        # Convert BGRA to RGB numpy array
        img = np.array(screenshot)
        rgb = img[:, :, :3][:, :, ::-1]  # BGRA -> RGB
        
        return rgb
    
    def capture_pil(self) -> Image.Image:
        """
        Capture screen as PIL Image for VLM processing.
        
        Returns:
            PIL Image in RGB format
        """
        rgb_array = self.capture_state()
        return Image.fromarray(rgb_array)
    
    def describe_screen(self, question: Optional[str] = None) -> str:
        """
        Get a description of what's currently on screen.
        
        Args:
            question: Optional specific question about the screen
            
        Returns:
            Text description of the screen content
        """
        if not self.vlm:
            return "[VLM not loaded - cannot see screen]"
        
        image = self.capture_pil()
        
        if question:
            prompt = question
        else:
            prompt = "Describe what you see on this computer screen. List the main UI elements, buttons, text fields, and any visible text."
        
        try:
            result = self.vlm.query(image, prompt)
            return result.get("answer", "[No response]")
        except Exception as e:
            logger.error(f"VLM query failed: {e}")
            return f"[Vision error: {e}]"
    
    def find_element(self, element_description: str) -> Optional[VisualElement]:
        """
        Find a UI element on screen and return its coordinates.
        
        Args:
            element_description: What to look for (e.g., "search box", "Submit button")
            
        Returns:
            VisualElement with (x, y) coordinates, or None if not found
        """
        if not self.vlm:
            logger.warning("VLM not loaded, using fallback coordinates")
            return None
        
        image = self.capture_pil()
        
        try:
            # Use Moondream2's point() function to locate elements
            result = self.vlm.point(image, element_description)
            points = result.get("points", [])
            
            if points:
                # Points are normalized (0-1), convert to screen coordinates
                point = points[0]  # Take first match
                x = int(point["x"] * self._screen_width)
                y = int(point["y"] * self._screen_height)
                
                logger.info(f"Found '{element_description}' at ({x}, {y})")
                return VisualElement(
                    description=element_description,
                    x=x,
                    y=y,
                    confidence=point.get("confidence", 1.0)
                )
            else:
                logger.warning(f"Element not found: {element_description}")
                return None
                
        except Exception as e:
            logger.error(f"Element detection failed: {e}")
            return None
    
    def find_all_elements(self, element_type: str) -> List[VisualElement]:
        """
        Find all instances of an element type on screen.
        
        Args:
            element_type: What to look for (e.g., "button", "text field")
            
        Returns:
            List of VisualElement objects
        """
        if not self.vlm:
            return []
        
        image = self.capture_pil()
        
        try:
            result = self.vlm.detect(image, element_type)
            objects = result.get("objects", [])
            
            elements = []
            for obj in objects:
                # Get center of bounding box
                x_min = obj.get("x_min", 0)
                x_max = obj.get("x_max", 0)
                y_min = obj.get("y_min", 0)
                y_max = obj.get("y_max", 0)
                
                x = int((x_min + x_max) / 2 * self._screen_width)
                y = int((y_min + y_max) / 2 * self._screen_height)
                
                elements.append(VisualElement(
                    description=element_type,
                    x=x,
                    y=y
                ))
            
            return elements
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def get_screen_dimensions(self) -> Tuple[int, int]:
        """Get current screen dimensions."""
        if self._screen_width == 0:
            self.capture_state()  # Force capture to get dimensions
        return self._screen_width, self._screen_height
    
    def answer_visual_question(self, question: str) -> str:
        """
        Answer any question about the current screen.
        
        Args:
            question: Question about the screen content
            
        Returns:
            Answer from the VLM
        """
        return self.describe_screen(question)
    
    def check_for_element(self, element_description: str) -> bool:
        """
        Check if an element exists on screen.
        
        Args:
            element_description: What to look for
            
        Returns:
            True if element is found
        """
        element = self.find_element(element_description)
        return element is not None
    
    def wait_for_element(self, element_description: str, timeout: float = 10.0, interval: float = 1.0) -> Optional[VisualElement]:
        """
        Wait for an element to appear on screen.
        
        Args:
            element_description: What to wait for
            timeout: Maximum wait time in seconds
            interval: Check interval in seconds
            
        Returns:
            VisualElement if found, None if timeout
        """
        import time
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            element = self.find_element(element_description)
            if element:
                return element
            time.sleep(interval)
        
        logger.warning(f"Timeout waiting for: {element_description}")
        return None
