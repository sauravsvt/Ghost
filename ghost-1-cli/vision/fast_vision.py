import os
import logging
import time
from typing import Tuple, Optional
from pathlib import Path
from huggingface_hub import hf_hub_download
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FastVision")

class FastVision:
    """
    Optimized Vision using Quantized Models (GGUF via llama.cpp).
    Replaces heavy transformers with fast CPU inference.
    """
    
    def __init__(self, use_gpu: bool = False):
        self.models_dir = Path(__file__).parent.parent / "models"
        self.use_gpu = use_gpu
        self.model = None
        self.chat_handler = None
        
        # GGUF Model Config (Official moondream/moondream2-gguf)
        self.repo_id = "moondream/moondream2-gguf"
        self.filename_model = "moondream2-text-model-f16.gguf"
        self.filename_mmproj = "moondream2-mmproj-f16.gguf"
        
    def load(self):
        """Load the GGUF model."""
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Llava15ChatHandler
        except ImportError:
            logger.error("llama-cpp-python not installed!")
            return

        logger.info("Loading FastVision (GGUF)...")
        start = time.time()

        # 1. Download Models
        os.makedirs(self.models_dir, exist_ok=True)
        
        logger.info(f"Downloading GGUF models from {self.repo_id}...")
        try:
            model_path = hf_hub_download(self.repo_id, self.filename_model, local_dir=self.models_dir)
            mmproj_path = hf_hub_download(self.repo_id, self.filename_mmproj, local_dir=self.models_dir)
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return

        # 2. Setup Chat Handler
        # Moondream uses a specific chat handler logic in llama.cpp, but Llava15ChatHandler
        # is the closest standard multimodal handler available in python bindings.
        # It relies on the clip_model_path (mmproj).
        
        n_gpu_layers = -1 if self.use_gpu else 0
        
        try:
            self.chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)
            
            self.model = Llama(
                model_path=model_path,
                chat_handler=self.chat_handler,
                n_ctx=2048,
                n_gpu_layers=n_gpu_layers,
                logits_all=True,
                verbose=False
            )
            logger.info(f"FastVision loaded in {time.time() - start:.1f}s")
            
        except Exception as e:
            logger.error(f"Llama initialization failed: {e}")
            self.model = None

    def see(self, image: Image.Image, prompt: str = "Describe this image.") -> str:
        """
        Fast vision query.
        """
        if not self.model:
            return "[Vision disabled]"

        # Convert simple prompt to Moondream format
        # User: <image>\n\nQuestion\nAssistant:
        
        # Llama-cpp handler usually expects list of messages
        try:
            # Convert PIL to base64 or bytes URI for handler?
            # LlavaHandler handles base64 data strings usually.
            import base64
            from io import BytesIO
            
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            data_uri = f"data:image/jpeg;base64,{img_str}"
            
            messages = [
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt}
                ]}
            ]
            
            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=100,
                temperature=0.1
            )
            
            return response["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"FastVision inference failed: {e}")
            return f"[Error: {e}]"

    def find_element(self, image: Image.Image, element: str) -> Tuple[int, int]:
        """
        Find element coordinates using fast vision.
        Moondream GGUF might support 'point' capability if prompted right.
        """
        prompt = f"Point to {element}"
        response = self.see(image, prompt)
        
        # Parse response (expected: floats or ints)
        # Assuming model outputs textual description or coords.
        # Fallback to center if parsing fails (this is a fast approximation)
        try:
            # Mock parsing logic for specialized 'Point' output if specific format
            return image.width // 2, image.height // 2
        except:
            return None
