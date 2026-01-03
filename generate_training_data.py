"""
Auto-generate training data to kickstart the agent
"""
import json

# Simulated training examples
training_examples = [
    {
        "ui": """Window: Desktop
[1] Button: 'Start'
[2] Edit: 'Type here to search'
[3] Button: 'Task View'
[4] Button: 'File Explorer'""",
        "goal": "Open the Start menu",
        "output": {"action": "click", "id": 1, "reason": "User wants to open Start menu"}
    },
    {
        "ui": """Window: Start Menu
[1] Edit: 'Search box'
[2] Button: 'All apps'
[3] ListItem: 'Chrome'
[4] ListItem: 'Notepad'
[5] ListItem: 'Calculator'""",
        "goal": "Open Notepad",
        "output": {"action": "click", "id": 4, "reason": "User wants Notepad"}
    },
    {
        "ui": """Window: Start Menu
[1] Edit: 'Search box'
[2] Button: 'All apps'""",
        "goal": "Search for Chrome",
        "output": {"action": "type", "id": 1, "text": "Chrome", "reason": "Search for Chrome"}
    },
    {
        "ui": """Window: Search Results
[1] ListItem: 'Google Chrome'
[2] ListItem: 'Chrome Remote Desktop'""",
        "goal": "Open Chrome browser",
        "output": {"action": "click", "id": 1, "reason": "First result is Chrome"}
    },
    {
        "ui": """Window: Google Chrome
[1] Edit: 'Address bar'
[2] Button: 'New Tab'
[3] Button: 'Bookmarks'""",
        "goal": "Go to YouTube",
        "output": {"action": "type", "id": 1, "text": "youtube.com", "reason": "Navigate to YouTube"}
    },
    {
        "ui": """Window: YouTube
[1] Edit: 'Search'
[2] Button: 'Trending'
[3] Button: 'Subscriptions'""",
        "goal": "Search for music",
        "output": {"action": "type", "id": 1, "text": "lofi music", "reason": "Search for lofi music"}
    },
    {
        "ui": """Window: Calculator
[1] Button: '1'
[2] Button: '2'
[3] Button: '3'
[4] Button: '+'
[5] Edit: 'Display'""",
        "goal": "Click number 2",
        "output": {"action": "click", "id": 2, "reason": "User wants to input 2"}
    },
    {
        "ui": """Window: Notepad
[1] Edit: 'Text editor'
[2] Button: 'File'
[3] Button: 'Edit'""",
        "goal": "Type hello world",
        "output": {"action": "type", "id": 1, "text": "Hello World", "reason": "Type text in editor"}
    },
    {
        "ui": """Window: File Explorer
[1] Edit: 'Address bar'
[2] Button: 'Back'
[3] Button: 'Forward'
[4] ListItem: 'Documents'
[5] ListItem: 'Downloads'""",
        "goal": "Open Documents folder",
        "output": {"action": "click", "id": 4, "reason": "Navigate to Documents"}
    },
    {
        "ui": """Window: Settings
[1] Button: 'System'
[2] Button: 'Personalization'
[3] Button: 'Apps'
[4] Edit: 'Find a setting'""",
        "goal": "Search for display settings",
        "output": {"action": "type", "id": 4, "text": "display", "reason": "Search settings"}
    }
]

print(f"[*] Generating {len(training_examples)} training examples...")

with open("dataset.jsonl", "w", encoding="utf-8") as f:
    for example in training_examples:
        f.write(json.dumps(example) + "\n")

print(f"[✓] Created dataset.jsonl with {len(training_examples)} examples")
print("[*] The agent now knows:")
print("  - How to click Start menu")
print("  - How to search and open apps")
print("  - How to navigate Chrome/YouTube")
print("  - How to use Calculator/Notepad")
print("\nReady for AI mode! Run: python main.py")
