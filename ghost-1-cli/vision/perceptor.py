"""
Ghost Vision v3.0 - Simplified Eye for Cloud Processing

Since Gemini handles all image understanding in the cloud,
this module just captures screenshots and returns bytes.
"""

import io
import logging
from PIL import Image

logger = logging.getLogger("Eye")


class Eye:
    """
    Simple screen capture for cloud API upload.
    
    Gemini handles the actual vision processing, so we just need
    to capture and compress the screenshot.
    """
    
    def __init__(self, max_resolution: tuple = (1920, 1080)):
        """
        Args:
            max_resolution: Maximum resolution to send (saves bandwidth)
        """
        self.max_resolution = max_resolution
        logger.info(f"Eye initialized (max res: {max_resolution})")
    
    def capture_raw(self) -> bytes:
        """
        Captures screen and returns JPEG bytes for API upload.
        
        Returns:
            JPEG image bytes
        """
        try:
            import pyautogui
            screenshot = pyautogui.screenshot()
        except Exception as e:
            # Fallback to mss if pyautogui fails
            import mss
            import numpy as np
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                sct_img = sct.grab(monitor)
                screenshot = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
        
        # Resize to save bandwidth (Gemini can handle any size, but faster with smaller)
        screenshot.thumbnail(self.max_resolution, Image.Resampling.LANCZOS)
        
        # Convert to JPEG bytes
        img_byte_arr = io.BytesIO()
        screenshot.save(img_byte_arr, format='JPEG', quality=85)
        
        logger.info(f"Captured screen: {screenshot.size}")
        return img_byte_arr.getvalue()
    
    def capture_region(self, x: int, y: int, width: int, height: int) -> bytes:
        """
        Capture a specific region of the screen.
        
        Useful for focusing on a particular window or element.
        """
        import pyautogui
        screenshot = pyautogui.screenshot(region=(x, y, width, height))
        
        img_byte_arr = io.BytesIO()
        screenshot.save(img_byte_arr, format='JPEG', quality=90)
        
        return img_byte_arr.getvalue()
    
    def get_change_percentage(self, before: bytes, after: bytes) -> float:
        """
        Calculate percentage of pixels that changed between two screenshots.
        Used for verifying if an action had visual effect.
        """
        import numpy as np
        
        img1 = Image.open(io.BytesIO(before))
        img2 = Image.open(io.BytesIO(after))
        
        # Ensure same size
        if img1.size != img2.size:
            img2 = img2.resize(img1.size)
        
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        
        # Calculate difference
        diff = np.abs(arr1.astype(float) - arr2.astype(float))
        change_pixels = np.sum(diff > 30)  # Threshold for "changed"
        total_pixels = arr1.size
        
        return (change_pixels / total_pixels) * 100


# Backwards compatibility alias
class ScreenPerceptor(Eye):
    """Alias for backwards compatibility with old code."""
    pass
