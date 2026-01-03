"""
Ghost Engine v5.0 (Local/Structure + Cheat Sheet Memory)
"""

import os
import json
import logging
from typing import Dict, Any, Optional
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from core.prompts import SYSTEM_PROMPT_STRUCTURE

logger = logging.getLogger("Brain")

class GhostEngine:
    """
    The Local Brain (Qwen 2.5 0.5B).
    Runs 100% locally on CPU with Cheat Sheet Memory.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        if Llama is None:
            raise ImportError("llama-cpp-python not installed. Run: pip install llama-cpp-python")
            
        # Find model if not provided
        if not model_path:
            model_path = self._find_model()
            
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"Model not found at {model_path}. Please download Qwen 2.5 0.5B GGUF to the 'models/' folder.")
            
        logger.info(f"Loading local brain: {model_path}")
        print(f"[*] Loading model: {os.path.basename(model_path)}...")
        
        # Initialize Llama
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096, 
            n_threads=4,  # Adjust based on CPU
            verbose=False
        )
        print("[*] Brain loaded.")

    def _find_model(self) -> Optional[str]:
        """Auto-discover .gguf model in models/ directory."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, "models")
        
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            return None
        
        # Priority: qwen models first, then any gguf
        files = os.listdir(models_dir)
        
        # First pass: look for qwen models
        for file in files:
            if "qwen" in file.lower() and file.endswith(".gguf"):
                return os.path.join(models_dir, file)
        
        # Second pass: any gguf that's not moondream (vision model)
        for file in files:
            if file.endswith(".gguf") and "moondream" not in file.lower():
                return os.path.join(models_dir, file)
        
        return None

    def load_memory(self) -> str:
        """
        Reads the Teacher's Cheat Sheet (dataset.jsonl).
        Returns formatted examples for few-shot learning.
        """
        examples = []
        dataset_path = "dataset.jsonl"
        
        if not os.path.exists(dataset_path):
            return ""
        
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    
                    # Format: Screen -> Goal -> Action
                    ui = data.get("ui", "")
                    goal = data.get("goal", "")
                    output = data.get("output", {})
                    
                    # Compact format
                    example = f"Goal: {goal}\nUI: {ui[:200]}...\nAction: {json.dumps(output)}"
                    examples.append(example)
        except Exception as e:
            logger.warning(f"Failed to load memory: {e}")
            return ""
        
        # Keep only last 5 examples to save context
        recent = examples[-5:]
        if recent:
            return "\n\n".join(recent)
        return ""

    def think(self, ui_tree: str, task: str) -> Dict[str, Any]:
        """
        Decision loop with Cheat Sheet Memory:
        UI Tree + Task + Past Examples -> Action JSON
        """
        prompt = self._build_prompt(ui_tree, task)
        
        # Generation
        output = self.llm(
            prompt,
            max_tokens=256,
            stop=["
