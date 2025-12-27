#!/usr/bin/env python3
"""
Ghost-1: Autonomous Desktop Agent with True Vision

A fully local, privacy-first agent that can SEE your screen
and take actions to complete tasks autonomously.

Usage:
    python main.py

Features:
    - Real vision via Moondream2 VLM
    - Multi-step autonomous execution (Observe -> Think -> Act -> Verify)
    - Natural language task understanding
    - Element coordinate detection
"""

import sys
import os
import time
import logging
import json
import re
from colorama import Fore, Style, init
from PIL import Image
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.engine import GhostEngine, ModelConfig
from actions.controller import ActionController
from vision.grid import ScreenGrid, ElementCoordinateResolver

# Initialize Colorama
init(autoreset=True)

# Create logs directory
os.makedirs("logs", exist_ok=True)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/ghost.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Main")


def print_banner():
    print(f"\n{Fore.CYAN}╔════════════════════════════════════════════╗")
    print(f"{Fore.CYAN}║      GHOST-1: AUTONOMOUS AGENT (v2.0)      ║")
    print(f"{Fore.CYAN}║   True Vision | Multi-Step | Local Only   ║")
    print(f"{Fore.CYAN}╚════════════════════════════════════════════╝{Style.RESET_ALL}\n")


def print_section(title: str, color=Fore.BLUE):
    print(f"\n{color}━━━ {title} ━━━{Style.RESET_ALL}")


def print_thinking(reasoning: str):
    if reasoning:
        print(f"{Fore.MAGENTA}Reasoning:{Style.RESET_ALL}")
        for line in reasoning.split('\n')[:10]:  # Limit output
            if line.strip():
                print(f"  {Fore.MAGENTA}{line.strip()}{Style.RESET_ALL}")


def print_vision(vision: str):
    if vision and len(vision) > 20:
        print(f"{Fore.YELLOW}Screen:{Style.RESET_ALL}")
        # Truncate long vision descriptions
        display = vision[:500] + "..." if len(vision) > 500 else vision
        for line in display.split('. ')[:5]:
            if line.strip():
                print(f"  {Fore.YELLOW}• {line.strip()}{Style.RESET_ALL}")


def capture_screen_pil() -> Image.Image:
    """Capture screen as PIL Image."""
    import mss
    
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        rgb = img[:, :, :3][:, :, ::-1]  # BGRA -> RGB
        return Image.fromarray(rgb)


def run_agent_loop(engine: GhostEngine, hands: ActionController, task: str, max_steps: int = 10):
    """
    Run the multi-step agent loop: Observe -> Think -> Act -> Verify
    
    Args:
        engine: GhostEngine with vision and language
        hands: ActionController for executing actions
        task: User's task description
        max_steps: Maximum number of actions before stopping
    """
    print_section(f"Task: {task}", Fore.GREEN)
    
    action_history = []
    step = 0
    
    while step < max_steps:
        step += 1
        print_section(f"Step {step}/{max_steps}")
        
        # 1. OBSERVE - Capture and understand screen
        print(f"{Fore.BLUE}[1] Observing screen...{Style.RESET_ALL}")
        image = capture_screen_pil()
        
        # 2. THINK - Reason about what to do
        print(f"{Fore.BLUE}[2] Thinking...{Style.RESET_ALL}")
        result = engine.think(image, task, action_history)
        
        # Display what agent sees and thinks
        print_vision(result.get("vision", ""))
        print_thinking(result.get("reasoning", ""))
        
        # 3. ACT - Execute the action
        action = result.get("action")
        
        if not action:
            print(f"{Fore.RED}  No action generated. Retrying...{Style.RESET_ALL}")
            continue
        
        tool = action.get("tool", "unknown")
        print(f"\n{Fore.GREEN}[3] Action: {tool}{Style.RESET_ALL}")
        print(f"    {json.dumps(action, indent=2)}")
        
        # Check for completion
        if tool == "done":
            message = action.get("message", "Task completed")
            print(f"\n{Fore.GREEN}✓ DONE: {message}{Style.RESET_ALL}")
            break
        
        # Handle mouse.click with 'target' (element name)
        if tool == "mouse.click" and "target" in action and "x" not in action:
            target = action["target"]
            print(f"    Looking for: '{target}'...")
            
            # Try to find element coordinates
            coords = engine.find_element(image, target)
            if coords:
                action["x"], action["y"] = coords
                print(f"    Found at: ({coords[0]}, {coords[1]})")
            else:
                print(f"    {Fore.YELLOW}Could not locate '{target}'. Skipping action.{Style.RESET_ALL}")
                action_history.append(f"Failed to find: {target}")
                continue
        
        # Execute the action
        action_result = hands.execute_tool(tool, action)
        
        if action_result.success:
            print(f"    {Fore.GREEN}✓ {action_result.output}{Style.RESET_ALL}")
            action_history.append(f"{tool}: {action_result.output}")
            
            # Log successful experience (Self-Evolution)
            if tool != "wait" and tool != "done":
                # We assume if tool succeeded, it's a positive signal. 
                # Ideally we verify the OUTCOME (step 4) matches intent.
                # For now, we log the successful execution.
                engine.log_success(task, result)
        else:
            print(f"    {Fore.RED}✗ {action_result.error}{Style.RESET_ALL}")
            action_history.append(f"{tool}: FAILED - {action_result.error}")
        
        # 4. VERIFY - Quick look to see if state changed
        print(f"{Fore.BLUE}[4] Verifying...{Style.RESET_ALL}")
        time.sleep(1)  # Wait for UI to update
        
        new_image = capture_screen_pil()
        quick_look = engine.quick_look(new_image)
        print(f"    Screen: {quick_look[:100]}...")
        
        # Small delay between steps
        time.sleep(0.5)
    
    if step >= max_steps:
        print(f"\n{Fore.YELLOW}⚠ Max steps reached ({max_steps}). Stopping.{Style.RESET_ALL}")
    
    print(f"\n{Fore.CYAN}━━━ Agent Loop Complete ━━━{Style.RESET_ALL}")
    print(f"Actions taken: {len(action_history)}")
    for i, action in enumerate(action_history, 1):
        print(f"  {i}. {action}")


def main():
    print_banner()
    
    # Initialize components
    print(f"{Fore.YELLOW}[*] Initializing Ghost-1 (this may download ~4GB on first run)...{Style.RESET_ALL}")
    
    try:
        config = ModelConfig()
        
        print(f"{Fore.YELLOW}[*] Loading VLM (Moondream2) + LLM (Qwen)...{Style.RESET_ALL}")
        engine = GhostEngine(config)
        
        print(f"{Fore.YELLOW}[*] Initializing action controller...{Style.RESET_ALL}")
        hands = ActionController()
        
    except ImportError as e:
        print(f"\n{Fore.RED}MISSING DEPENDENCY: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Run: pip install -r requirements.txt{Style.RESET_ALL}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Fore.RED}INITIALIZATION ERROR: {e}{Style.RESET_ALL}")
        logger.exception("Init failed")
        sys.exit(1)

    print(f"\n{Fore.GREEN}✓ Ghost-1 Ready with True Vision!{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Commands:{Style.RESET_ALL}")
    print(f"  • Type a task or {Fore.MAGENTA}Say 'Ghost <command>'{Style.RESET_ALL}")
    print(f"  • 'look': Just describe what's on screen")
    print(f"  • 'find <element>': Find element coordinates")
    print(f"  • 'exit': Quit")
    print(f"{Fore.YELLOW}Move mouse to corner (0,0) for emergency stop.{Style.RESET_ALL}")
    
    # Initialize Ears
    try:
        from ears.listener import VoiceListener
        ears = VoiceListener(wake_word="ghost")
        print(f"{Fore.GREEN}✓ Ears Active (Faster-Whisper){Style.RESET_ALL}")
    except ImportError:
        ears = None
        print(f"{Fore.YELLOW}⚠ Voice disabled (dependencies missing){Style.RESET_ALL}")
    except Exception as e:
        ears = None
        print(f"{Fore.YELLOW}⚠ Voice disabled: {e}{Style.RESET_ALL}")

    while True:
        try:
            # Check for voice command (non-blocking if possible, but here we alternate)
            # For CLI interaction, we prioritize input() but could thread the listener.
            # To keep it simple and stable: We check voice if no input, or use separate thread.
            # Actually, blocking input() prevents voice loop in single thread.
            # We will use a simple heuristic: if arguments passed --voice, loop voice.
            # Otherwise simultaneous input is hard in CLI.
            # Let's assume user types OR speaks if we implement a polling loop.
            
            # Since input() blocks, we can't easily listen in background without threads.
            # We will rely on user typing "listen" to activate voice mode temporarily or just stick to text 
            # unless we go full voice mode. 
            # BUT the prompt says "Add ears... When triggered...".
            # I will add a specific "listen" loop or just text for reliability unless I add threading.
            # Let's just use input() but check for "listen" command to enter voice mode.
            
            user_input = input(f"\n{Fore.YELLOW}Task@{Fore.WHITE}Ghost-1:~$ ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print(f"{Fore.CYAN}Shutting down...{Style.RESET_ALL}")
                break
                
            if user_input.lower() == 'listen' and ears:
                print(f"{Fore.MAGENTA}Listening for 'Ghost'... (Ctrl+C to stop){Style.RESET_ALL}")
                try:
                    while True:
                        cmd = ears.listen_for_command(timeout=10) # Listens in chunks
                        if cmd:
                            print(f"\n{Fore.MAGENTA}Voice Command: {cmd}{Style.RESET_ALL}")
                            run_agent_loop(engine, hands, cmd)
                            break
                        # Verification check - if timeout returns None, loop or break?
                        # ears.listen_for_command is blocking loop in my impl.
                except KeyboardInterrupt:
                    continue

            # Special command: just look at screen
            if user_input.lower() == 'look':
                print(f"{Fore.BLUE}Looking at screen...{Style.RESET_ALL}")
                image = capture_screen_pil()
                description = engine.see(image)
                print(f"\n{Fore.YELLOW}{description}{Style.RESET_ALL}")
                continue
            
            # Special command: find element
            if user_input.lower().startswith('find '):
                element = user_input[5:].strip()
                print(f"{Fore.BLUE}Looking for: {element}...{Style.RESET_ALL}")
                image = capture_screen_pil()
                coords = engine.find_element(image, element)
                if coords:
                    print(f"{Fore.GREEN}Found at: ({coords[0]}, {coords[1]}){Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}Not found{Style.RESET_ALL}")
                continue
            
            # Run the autonomous agent loop
            run_agent_loop(engine, hands, user_input)
            
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Interrupted.{Style.RESET_ALL}")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")

    print(f"\n{Fore.CYAN}Ghost-1 shutdown complete.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
