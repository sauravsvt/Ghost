"""
Ghost Engine - Multimodal Vision-Language Inference

This module provides the brain for Ghost-1, combining:
1. Vision: Moondream2 VLM for screen understanding
2. Language: Qwen-1.5 for reasoning and action generation

The engine now truly "sees" the screen.
"""

import os
import sys
import time
import logging
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
import numpy as np
from PIL import Image

logger = logging.getLogger("GhostEngine")


@dataclass
class ModelConfig:
    """Configuration for Ghost inference engine."""
    
    # Text LLM (for reasoning/action generation)
    llm_model_repo: str = "Qwen/Qwen1.5-1.8B-Chat-GGUF"
    llm_model_file: str = "qwen1_5-1_8b-chat-q4_k_m.gguf"
    
    # Vision LLM (for screen understanding)
    vlm_model: str = "vikhyatk/moondream2"
    vlm_revision: str = "2025-06-21"
    
    # Paths
    models_dir: str = field(default_factory=lambda: str(Path(__file__).parent.parent / "models"))
    
    # Inference params
    context_length: int = 4096
    max_tokens: int = 1024
    temperature: float = 0.3
    
    # Hardware
    use_gpu: bool = False
    n_threads: int = 0
    
    @property
    def llm_path(self) -> str:
        return os.path.join(self.models_dir, self.llm_model_file)


def download_llm(config: ModelConfig) -> str:
    """Download text LLM if not present."""
    from huggingface_hub import hf_hub_download
    
    model_path = config.llm_path
    if os.path.exists(model_path):
        return model_path
    
    os.makedirs(config.models_dir, exist_ok=True)
    logger.info(f"Downloading LLM: {config.llm_model_file}...")
    
    return hf_hub_download(
        repo_id=config.llm_model_repo,
        filename=config.llm_model_file,
        local_dir=config.models_dir,
        local_dir_use_symlinks=False
    )


class GhostEngine:
    """
    The Brain - Multimodal Vision-Language Agent.
    
    Combines:
    - VLM (Moondream2) for visual understanding
    - LLM (Qwen) for reasoning and action generation
    
    The agent can now truly see the screen and reason about it.
    """
    
    SYSTEM_PROMPT = """You are Ghost-1, an autonomous desktop agent that can SEE the screen and execute actions.

You will receive:
1. A description of what is currently visible on screen (from your vision system)
2. The user's task

Based on what you SEE, decide what action to take.

OUTPUT FORMAT:
<think>
[Your reasoning based on what you see on screen]
</think>
{"tool": "tool_name", "param": "value"}

TOOLS:
- browser.open: {"tool": "browser.open", "url": "https://youtube.com"}
- keyboard.type: {"tool": "keyboard.type", "text": "search query"}  
- keyboard.hotkey: {"tool": "keyboard.hotkey", "keys": "ctrl+t"}
- mouse.click: {"tool": "mouse.click", "target": "search button"} - Use 'target' for element names
- wait: {"tool": "wait", "seconds": 2}
- done: {"tool": "done", "message": "Task completed"}

RULES:
- Use vision descriptions to understand the current state
- Output exactly ONE action at a time
- Use element names in mouse.click target (e.g., "search button", "play button")
- Say 'done' when the task is complete"""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.llm = None
        self.vlm = None
        self.memory = None
        self._load_engines()

    def _load_engines(self):
        """Load both VLM and LLM."""
        logger.info("Initializing Ghost Engine (Vision + Language)...")
        start = time.time()
        
        # Load VLM (Moondream2)
        self._load_vlm()
        
        # Load LLM (Qwen)
        self._load_llm()
        
        # Load Memory (Experience Replay)
        try:
            from core.trainer import ExperienceManager
            self.memory = ExperienceManager()
            logger.info("Experience Memory loaded.")
        except ImportError:
            logger.warning("ExperienceManager not found (trainer.py missing?). Learning disabled.")
            self.memory = None

        logger.info(f"Engines ready in {time.time() - start:.1f}s")
    
    def log_success(self, task: str, result: Dict[str, Any]):
        """Log a successful interaction to Experience Replay."""
        if self.memory and result.get("action"):
            self.memory.add_experience(
                task=task,
                vision_desc=result.get("vision", ""),
                reasoning=result.get("reasoning", ""),
                action=result.get("action")
            )
    
    @staticmethod
    def absmean_quantization(W: np.ndarray) -> np.ndarray:
        """
        Simulate BitNet 1.58-bit Absmean Quantization.
        Formula: W ~ Round(gamma * W), gamma = 1 / mean(|W|)
        
        This process constrains weights to {-1, 0, 1} for high-speed inference.
        """
        eps = 1e-5
        gamma = 1.0 / (np.mean(np.abs(W)) + eps)
        W_scaled = W * gamma
        W_quant = np.round(W_scaled)
        W_quant = np.clip(W_quant, -1, 1)
        return W_quant

    def _load_vlm(self):
        """Load Vision-Language Model (FastVision GGUF preferred)."""
        # Try FastVision (GGUF via llama.cpp) first
        try:
            from vision.fast_vision import FastVision
            logger.info("Initializing FastVision (Optimized GGUF)...")
            self.vlm = FastVision(use_gpu=self.config.use_gpu)
            self.vlm.load()
            if self.vlm.model:
                logger.info("FastVision loaded successfully.")
                return
        except Exception as e:
            logger.warning(f"FastVision loading failed: {e}")
            logger.info("Falling back to Standard VLM (Transformers)...")

        # Fallback to Standard Transformers
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info("Loading Moondream2 VLM (Standard)...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.vlm_model,
                revision=self.config.vlm_revision,
                trust_remote_code=True
            )
            
            # Load model without device_map (works better on CPU)
            self.vlm = AutoModelForCausalLM.from_pretrained(
                self.config.vlm_model,
                revision=self.config.vlm_revision,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Move to appropriate device
            device = "cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu"
            self.vlm = self.vlm.to(device)
            self.vlm.eval()  # Set to evaluation mode
            
            logger.info(f"VLM loaded on {device}")
            
        except Exception as e:
            logger.warning(f"VLM loading failed: {e}")
            logger.warning("Vision will be disabled.")
            self.vlm = None
            self.tokenizer = None
    
    def _load_llm(self):
        """Load Qwen LLM for reasoning."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("Install: pip install llama-cpp-python huggingface-hub")
        
        model_path = download_llm(self.config)
        logger.info(f"Loading LLM: {model_path}")
        
        n_threads = self.config.n_threads
        if n_threads == 0:
            import multiprocessing
            n_threads = max(1, multiprocessing.cpu_count() // 2)
        
        self.llm = Llama(
            model_path=model_path,
            n_ctx=self.config.context_length,
            n_threads=n_threads,
            verbose=False
        )
        
        logger.info("LLM loaded")
    
    def see(self, image: Image.Image, question: Optional[str] = None) -> str:
        """
        Use VLM to understand what's on screen.
        
        Args:
            image: PIL Image of the screen
            question: Optional specific question
            
        Returns:
            Description of screen content
        """
        if not self.vlm:
            return "[Vision disabled - VLM not loaded]"
        
        if question is None:
            question = "Describe this computer screen. What application is open? What buttons, text fields, and clickable elements do you see? Be specific about their locations (top, bottom, left, right, center)."
        
        try:
            # Moondream 2025-06-21 API: query()
            result = self.vlm.query(image, question)
            return result.get("answer", str(result))
        except Exception as e:
            logger.error(f"Vision failed: {e}")
            return f"[Vision error: {e}]"
    
    def find_element(self, image: Image.Image, element: str) -> Optional[Tuple[int, int]]:
        """
        Find an element's coordinates on screen.
        
        Args:
            image: PIL Image of the screen
            element: What to find (e.g., "search button")
            
        Returns:
            (x, y) coordinates or None
        """
        if not self.vlm:
            return None
        
        try:
            # Moondream 2025-06-21 API: point()
            result = self.vlm.point(image, element)
            points = result.get("points", [])
            
            if points:
                # Get image dimensions
                width, height = image.size
                point = points[0]
                x = int(point.get("x", 0.5) * width)
                y = int(point.get("y", 0.5) * height)
                logger.info(f"VLM found '{element}' at ({x}, {y})")
                return x, y
                
        except Exception as e:
            logger.error(f"Element detection failed: {e}")
        
        return None
    
    def think(self, image: Image.Image, user_task: str, previous_actions: List[str] = None) -> Dict[str, Any]:
        """
        Main reasoning method - combines vision and language.
        
        Args:
            image: PIL Image of screen
            user_task: What the user wants to do
            previous_actions: List of actions already taken
            
        Returns:
            Dict with 'vision', 'reasoning', 'action' keys
        """
        result = {
            "vision": "",
            "reasoning": "",
            "action": None,
            "raw_response": ""
        }
        
        # Step 1: See the screen
        logger.info("Looking at screen...")
        vision_description = self.see(image)
        result["vision"] = vision_description
        
        # Step 2: Build context for LLM
        context = f"""CURRENT SCREEN:
{vision_description}

USER TASK: {user_task}"""

        if previous_actions:
            context += f"\n\nACTIONS ALREADY TAKEN:\n" + "\n".join(f"- {a}" for a in previous_actions[-5:])
        
        prompt = f"""<|im_start|>system
{self.SYSTEM_PROMPT}
<|im_end|>
<|im_start|>user
{context}
<|im_end|>
<|im_start|>assistant
<think>
"""
        
        # Step 3: Generate reasoning and action
        logger.info("Thinking...")
        
        if not self.llm:
            result["action"] = {"tool": "done", "message": "LLM not loaded"}
            return result
        
        try:
            output = self.llm(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                stop=["<|im_end|>", "<|im_start|>"]
            )
            
            response = output["choices"][0]["text"]
            result["raw_response"] = "<think>\n" + response
            
            # Extract reasoning
            if "</think>" in response:
                reasoning = response.split("</think>")[0].strip()
                result["reasoning"] = reasoning
            
            # Extract JSON action
            json_match = re.search(r'\{[^{}]*"tool"[^{}]*\}', response, re.DOTALL)
            if json_match:
                try:
                    action = json.loads(json_match.group())
                    result["action"] = action
                except json.JSONDecodeError:
                    pass
            
            # If action has 'target', try to resolve coordinates
            if result["action"] and "target" in result["action"]:
                target = result["action"]["target"]
                coords = self.find_element(image, target)
                if coords:
                    result["action"]["x"] = coords[0]
                    result["action"]["y"] = coords[1]
                    logger.info(f"Resolved '{target}' to {coords}")
            
        except Exception as e:
            logger.error(f"Thinking failed: {e}")
            result["action"] = {"tool": "done", "message": f"Error: {e}"}
        
        return result
    
    def quick_look(self, image: Image.Image) -> str:
        """Quick screen description for verification."""
        if not self.vlm:
            return "[No vision]"
        
        try:
            # Moondream 2025-06-21 API: caption()
            result = self.vlm.caption(image, length="short")
            return result.get("caption", str(result))
        except Exception as e:
            return f"[Vision error: {e}]"

