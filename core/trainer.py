"""
Ghost Trainer - Self-Evolution Module
Implements Experience Replay and Background Training (LoRA).

Concept:
1. Experience Replay: Save successful (Context -> Action) pairs.
2. Background Training: Fine-tune a lightweight LoRA adapter on these pairs.
"""

import os
import json
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Try imports for training (optional at runtime if just inferencing)
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False

logger = logging.getLogger("GhostTrainer")

@dataclass
class TrainingConfig:
    model_name: str = "Qwen/Qwen1.5-1.8B-Chat" # Base model for LoRA
    adapter_path: str = "models/ghost_adapter"
    experience_file: str = "logs/experience.json"
    batch_size: int = 1
    learning_rate: float = 2e-4
    epochs: int = 3

class ExperienceManager:
    """Manages the agent's memory of successful actions."""
    
    def __init__(self, filepath: str = "logs/experience.json"):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
    def add_experience(self, task: str, vision_desc: str, reasoning: str, action: Dict[str, Any], outcome: str = "success"):
        """Save a successful interaction."""
        entry = {
            "timestamp": time.time(),
            "task": task,
            "vision": vision_desc,
            "reasoning": reasoning,
            "action": action,
            "outcome": outcome
        }
        
        data = self.load_experience()
        data.append(entry)
        
        with open(self.filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Experience saved: {task} -> {action.get('tool')}")

    def load_experience(self) -> List[Dict]:
        if not os.path.exists(self.filepath):
            return []
        try:
            with open(self.filepath, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []

class GhostTrainer:
    """Background Trainer using LoRA."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.experience = ExperienceManager(config.experience_file)
        
    def train(self):
        """Run a fine-tuning session on collected experience."""
        if not TRAINING_AVAILABLE:
            logger.error("Training dependencies missing. Install: pip install peft datasets scikit-learn transformers torch")
            return
            
        data = self.experience.load_experience()
        if not data:
            logger.warning("No experience to train on.")
            return

        logger.info(f"Starting background training on {len(data)} samples...")
        
        # 1. Prepare Data
        # Format: User Prompt (Vision+Task) -> Assistant Response (Reasoning+Action)
        formatted_data = []
        for item in data:
            prompt = f"Vision: {item['vision']}\nTask: {item['task']}\n"
            response = f"<think>{item['reasoning']}</think>\n{json.dumps(item['action'])}"
            text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
            formatted_data.append({"text": text})
            
        dataset = Dataset.from_list(formatted_data)
        
        # 2. Load Model (Base)
        logger.info(f"Loading base model: {self.config.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name, 
            trust_remote_code=True,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # 3. Apply LoRA (Logic: 1.58-bit not supported in training yet, use LoRA)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=8, 
            lora_alpha=32, 
            lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        # 4. Tokenize
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
            
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        
        # 5. Train
        training_args = TrainingArguments(
            output_dir=self.config.adapter_path,
            per_device_train_batch_size=self.config.batch_size,
            num_train_epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            save_steps=100,
            logging_steps=10,
            use_cpu=not torch.cuda.is_available()
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        
        trainer.train()
        
        # 6. Save Adapter
        model.save_pretrained(self.config.adapter_path)
        logger.info(f"Training complete. Adapter saved to {self.config.adapter_path}")

if __name__ == "__main__":
    # Test Experience Manager
    exp = ExperienceManager()
    exp.add_experience("Test Task", "Screen shows desktop", "Planning test", {"tool": "test"})
    print("Experience saved.")
