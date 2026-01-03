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

logger = logging.getLogger("Brain")

# Updated System Prompt with Tool Definitions
SYSTEM_PROMPT_WITH_TOOLS = """You are Ghost-1, a Windows UI Automation Agent.
You read UI elements and output JSON commands. Follow these rules EXACTLY.

═══════════════════════════════════════════
AVAILABLE ACTIONS (USE THESE EXACT NAMES):
═══════════════════════════════════════════

1. CLICK a button/element by ID:
   {"action": "click", "id": 12}
   
2. TYPE text:
   {"action": "type", "id": 5, "text": "hello"}

3. SCROLL down/up:
   {"action": "scroll", "direction": "down", "amount": 3}
   {"action": "scroll.down"}  // shorthand for scroll down
   {"action": "scroll.up"}    // shorthand for scroll up

4. LAUNCH an app:
   {"action": "sys.launch", "app": "calculator"}
   Apps: calculator, notepad, paint, chrome, edge, explorer

5. WAIT:
   {"action": "wait"}

6. DONE:
   {"action": "done"}

7. MACRO ACTIONS (CALM MODE):
   You can output a LIST of actions to execute in sequence:
   [
     {"action": "sys.launch", "app": "notepad"},
     {"action": "wait"},
     {"action": "type", "text": "hello"}
   ]

═══════════════════════════════════════════
CRITICAL RULES:
═══════════════════════════════════════════

- To CLICK a button with ID 24: {"action": "click", "id": 24}
- To SCROLL down: {"action": "scroll.down"} or {"action": "scroll", "direction": "down"}
- To OPEN calculator: {"action": "sys.launch", "app": "calculator"}
- THERE IS NO "open" ACTION! Use "click" for IDs, "sys.launch" for apps
- If you don't see the element you need, try scrolling first
- Only use IDs that exist in the UI Tree
- Output JSON only, no markdown
- Use MACRO ACTIONS (List) for multiple steps (e.g. Launch -> Wait -> Type)
"""


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

    
    def think(self, ui_tree: str, task: str) -> Any:
        """
        Decision loop with TRM-Inspired Recursive Reasoning.
        """
        return self._think_recursively(ui_tree, task, depth=1)

    def _think_recursively(self, ui_tree: str, task: str, depth: int = 1) -> Any:
        """
        Ghost-TRM: Recurse on thoughts to refine action.
        Z_t+1 = f(Z_t)
        """
        # Step 1: Initial Thought (Z_0)
        prompt = self._build_prompt(ui_tree, task)
        prompt += "\n\nInitial Thought (Draft 1):"
        
        # Draft 1
        output_1 = self.llm(prompt, max_tokens=256, stop=["\n", "```"], temperature=0.3)
        draft_1 = output_1["choices"][0]["text"].strip()
        
        if depth == 0:
            return self._parse_json(draft_1)

        # Step 2: Recursion / Critique (Z_1)
        # We feed draft 1 back and ask for refinement
        prompt += f" {draft_1}\n\nReview the above. If correct, repeat it. If wrong, correct it.\nJSON ONLY:"
        
        output_2 = self.llm(prompt, max_tokens=256, stop=["\n", "```"], temperature=0.1)
        final_response = output_2["choices"][0]["text"].strip()
        
        logger.info(f"🧠 TRM: {draft_1} -> {final_response}")
        
        # Robust Parsing with Fallback to System 1
        result = self._parse_json(final_response)
        if result.get("action") == "wait" and result.get("reason") == "Parse error":
             logger.warning("TRM Refinement failed (Parse Error). Falling back to Draft 1 (System 1).")
             return self._parse_json(draft_1)
             
        return result

    def _parse_json(self, text: str) -> Any:
        """Robust JSON parsing for Dict or List, with Regex fallback."""
        try:
            # 1. Clean Markdown
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            return json.loads(text)
        except json.JSONDecodeError:
            # 2. Regex Fallback (Find first outer valid JSON object/list)
            import re
            try:
                # Look for {...} or [...]
                match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
                if match:
                    return json.loads(match.group(1))
            except:
                pass
                
            logger.error(f"Failed to parse LLM output. Text: {text}")
            return {"action": "wait", "reason": "Parse error"}
    
    def analyze_ui(self, ui_tree: str) -> Dict[str, Any]:
        """
        UNIVERSAL INTENT: Phase 3 Autonomy.
        Asks the Brain: "What is this app? What should I do here?"
        """
        prompt = self._build_prompt(ui_tree, "Identify Application and Generate 3 Exploration Goals")
        prompt += "\n\nAnalyze the UI ABOVE. Output JSON:\n"
        prompt += '{\n  "app_type": "Brief Type (e.g. Text Editor)",\n  "suggested_goals": ["Goal 1", "Goal 2", "Goal 3"]\n}'
        prompt += "\nJSON ONLY:"
        
        output = self.llm(prompt, max_tokens=256, stop=["\n", "```"], temperature=0.1)
        text = output["choices"][0]["text"].strip()
        
        logger.info(f"🧠 Universal Intent Raw Output: {text}")
        
        # Try to parse
        result = self._parse_json(text)
        
        # Fallback if parsing completely fails and returns the default error dict
        if result.get("action") == "wait" and result.get("reason") == "Parse error":
             logger.error("Failed to parse Intent. Converting raw text to generic goal.")
             # Last resort: Try to extract *anything* that looks like a goal from the text
             return {
                 "app_type": "Unknown",
                 "suggested_goals": ["Explore interface"]
             }
             
        return result

    def _build_prompt(self, ui_tree: str, task: str) -> str:
        """Build the full prompt with system + memory + current task."""
        memory = self.load_memory()
        
        parts = [SYSTEM_PROMPT_WITH_TOOLS]
        
        if memory:
            parts.append(f"\n### CHEAT SHEET (Recent Examples)\n{memory}\n")
        
        parts.append(f"\n### CURRENT TASK\nGoal: {task}\n\nUI Tree:\n{ui_tree}\n\nYour Decision (JSON):")
        
        return "".join(parts)


