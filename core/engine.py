"""
Ghost Engine v5.0 (Local/Structure)
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
    Runs 100% locally on CPU.
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
        # n_ctx=2048 should be enough for UI trees
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
            
        for file in os.listdir(models_dir):
            if file.endswith(".gguf"):
                return os.path.join(models_dir, file)
        
        return None

    def think(self, ui_tree: str, task: str) -> Dict[str, Any]:
        """
        Decision loop: UI Tree + Task -> Action JSON
        """
        prompt = self._build_prompt(ui_tree, task)
        
        # Generation
        output = self.llm(
            prompt,
            max_tokens=256,
            stop=["<|endoftext|>", "User:", "###"],
            echo=False,
            temperature=0.1  # Low temp for deterministic logic
        )
        
        text = output['choices'][0]['text'].strip()
        logger.debug(f"Raw LLM output: {text}")
        
        return self._parse_json(text)

    def _build_prompt(self, ui_tree: str, task: str) -> str:
        """Construct Qwen/ChatML prompt."""
        # Simple ChatML format or similar
        return f"""<|im_start|>system
{SYSTEM_PROMPT_STRUCTURE}<|im_end|>
<|im_start|>user
Goal: {task}

Current UI:
{ui_tree}<|im_end|>
<|im_start|>assistant
"""

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """extract JSON from text."""
        try:
            # Find first { and last }
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                json_str = text[start:end+1]
                return json.loads(json_str)
        except Exception as e:
            logger.error(f"JSON Parse error: {e}")
        
        # Fallback
        return {"action": "wait", "reason": "Failed to parse brain output"}
