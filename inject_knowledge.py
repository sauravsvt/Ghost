import json
import random
from colorama import Fore, Style, init

# CONFIGURATION
OUTPUT_FILE = "dataset_knowledge.jsonl"
USE_REAL_DATASET = False # Set to True to download 2GB+ "osunlp/Mind2Web"

# Synthetic Mind2Web Data (High-Quality Reasoning Traces)
SYNTHETIC_DATA = [
    {
        "confirmed_task": "Search for 'Mechanical Keyboard' on Amazon",
        "action_reprs": ["Type 'Mechanical Keyboard' -> Click Search"],
        "browser_html": [
             {"tag": "input", "backend_node_id": "101", "name": "field-keywords", "value": ""},
             {"tag": "button", "backend_node_id": "102", "text": "Go", "type": "submit"},
             {"tag": "div", "backend_node_id": "103", "text": "Amazon Logo"},
             {"tag": "a", "backend_node_id": "104", "text": "Cart"}
        ],
        "target": "101",
        "action_type": "type",
        "value": "Mechanical Keyboard"
    },
    {
        "confirmed_task": "Select 'Price: Low to High' from sort menu",
        "action_reprs": ["Click Sort -> Select Low to High"],
        "browser_html": [
             {"tag": "select", "backend_node_id": "201", "name": "s-result-sort-select"},
             {"tag": "option", "backend_node_id": "202", "text": "Featured"},
             {"tag": "option", "backend_node_id": "203", "text": "Price: Low to High"},
             {"tag": "option", "backend_node_id": "204", "text": "Price: High to Low"}
        ],
        "target": "203",
        "action_type": "click",
        "value": ""
    },
     {
        "confirmed_task": "Click 'Add to Cart' for the visible item",
        "action_reprs": ["Click Add to Cart"],
        "browser_html": [
             {"tag": "div", "backend_node_id": "301", "text": "Keychron K2"},
             {"tag": "span", "backend_node_id": "302", "text": "$79.99"},
             {"tag": "button", "backend_node_id": "303", "text": "Add to Cart"},
             {"tag": "button", "backend_node_id": "304", "text": "Buy Now"}
        ],
        "target": "303",
        "action_type": "click",
        "value": ""
    }
]

def format_uia_tree(dom_elements):
    """
    Transmutes Web DOM elements into Ghost-1 UIA Tree format.
    """
    tree_lines = []
    target_map = {} 
    
    tree_lines.append(f"Window: Browser Content")
    
    for i, elem in enumerate(dom_elements):
        tag = elem.get('tag', 'control')
        text = elem.get('text', '') or elem.get('value', '') or elem.get('name', '')
        
        if tag in ['button', 'submit', 'a']: control_type = "Button"
        elif tag in ['input', 'textarea']: control_type = "Edit"
        elif tag in ['select', 'option']: control_type = "ComboBox"
        elif tag in ['div', 'span', 'p']: control_type = "Text"
        else: control_type = "Pane"
        
        fake_id = i + 10 # Offset
        clean_text = text.replace('\n', ' ').strip()[:50]
        
        if not clean_text and control_type == "Pane": continue

        line = f"[{fake_id}] {control_type}: '{clean_text}'"
        tree_lines.append(line)
        target_map[elem['backend_node_id']] = fake_id
        
    return "\n".join(tree_lines), target_map

def main():
    init()
    print(f"{Fore.CYAN}╔══════════════════════════════════════╗")
    print(f"║   GHOST-1: KNOWLEDGE INJECTION       ║")
    print(f"║   Source: osunlp/Mind2Web (Hybrid)   ║")
    print(f"╚══════════════════════════════════════╝{Style.RESET_ALL}")

    if USE_REAL_DATASET:
        print(f"{Fore.YELLOW}[*] Downloading Mind2Web...{Style.RESET_ALL}")
        from datasets import load_dataset
        dataset = load_dataset("osunlp/Mind2Web", split="train", streaming=True, trust_remote_code=True)
        # ... logic to iterate real dataset ...
    else:
        print(f"{Fore.YELLOW}[*] Using SYNTHETIC Knowledge (Fast Mode)...{Style.RESET_ALL}")
        dataset = SYNTHETIC_DATA

    count = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sample in dataset:
            try:
                # 1. TRANSMUTE
                instruction = sample['confirmed_task']
                dom = sample['browser_html']
                real_target_id = sample['target']
                
                tree, id_map = format_uia_tree(dom)
                
                # Find the mapped ID
                ghost_id = id_map.get(real_target_id)
                if not ghost_id: continue
                
                # 2. GENERATE OUTPUT
                if sample['action_type'] == 'type':
                    output = {"action": "type", "id": ghost_id, "text": sample['value'], "reason": f"Steps: {sample['action_reprs'][0]}"}
                else:
                    output = {"action": "click", "id": ghost_id, "reason": f"Steps: {sample['action_reprs'][0]}"}
                
                # 3. SAVE
                entry = {
                    "ui": tree,
                    "goal": instruction,
                    "output": output
                }
                
                f.write(json.dumps(entry) + "\n")
                count += 1
                
            except Exception as e:
                print(f"Error: {e}")
                continue

    print(f"{Fore.GREEN}[+] Transmuted {count} lessons into {OUTPUT_FILE}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
