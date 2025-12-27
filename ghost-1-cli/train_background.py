"""
Ghost-1 Background Trainer
Periodically run this script to fine-tune the agent on your own usage patterns.
Uses GRPO-style experience replay with LoRA adapters.
"""

import sys
import os
import logging
from core.trainer import GhostTrainer, TrainingConfig

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BackgroundTrainer")

def main():
    print("Ghost-1 Self-Evolution: Background Trainer")
    print("==========================================")
    
    # Check for experience logs
    if not os.path.exists("logs/experience.json"):
        print("No experience logs found (logs/experience.json).")
        print("Use the agent to solve tasks first!")
        return

    print("Initializing Trainer (this requires PyTorch + PEFT + Transformers)...")
    
    config = TrainingConfig(
        model_name="Qwen/Qwen1.5-1.8B-Chat", # Using PyTorch base model for training
        adapter_path="models/ghost_adapter",
        experience_file="logs/experience.json",
        epochs=3
    )
    
    trainer = GhostTrainer(config)
    
    try:
        trainer.train()
        print("\nSUCCESS: Adapter updated at models/ghost_adapter")
        print("The agent will now perform better on tasks it has seen before.")
    except Exception as e:
        print(f"\nTraining Failed: {e}")
        print("Ensure you have 'peft', 'transformers', 'torch' installed.")

if __name__ == "__main__":
    main()
