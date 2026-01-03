# Ghost-1 Architecture

## Overview

Ghost-1 is a local-first, autonomous UI automation agent that runs entirely on your machine using small language models (0.5B-2B parameters).

## System Architecture

```
┌─────────────────────────────────────────────┐
│              User Request                    │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│          Main Loop (main.py)                 │
│  • CLI parsing                              │
│  • Logging setup                            │
│  • Task orchestration                       │
└──────────┬────────────────┬─────────────────┘
           │                │
           ▼                ▼
    ┌──────────┐     ┌──────────────┐
    │  Brain   │     │    Eyes      │
    │  Engine  │     │  UIReader    │
    └──────────┘     └──────────────┘
           │                │
           │                │
           ▼                ▼
    ┌──────────────────────────────┐
    │    Hands (ActionController)  │
    │  • Click                     │
    │  • Type                      │
    │  • Scroll                    │
    │  • Launch apps               │
    └──────────────────────────────┘
                   │
                   ▼
            ┌────────────┐
            │   System   │
            │  (Windows) │
            └────────────┘
```

## Core Components

### 1. Vision Layer (`vision/`)

**UIReader** (`vision/structure.py`)
- Scans active window using pywinauto (UIA backend)
- Extracts UI element tree (buttons, text fields, etc.)
- Assigns unique IDs to each element
- Returns structured text representation

**Key Methods:**
- `capture_tree()` - Get current UI state
- `click_id(id)` - Click element by ID
- `type_in_id(id, text)` - Type into element

**Technology:** pywinauto with UIA (UI Automation) backend

### 2. Brain Layer (`core/`)

**GhostEngine** (`core/engine.py`)
- Local LLM inference using llama-cpp-python
- Processes UI tree + user goal → generates action JSON
- Uses "Cheat Sheet Memory" (few-shot learning from `dataset.jsonl`)
- Temperature: 0.2 for deterministic behavior

**Prompt Structure:**
```
System Prompt (tools + rules)
+
Cheat Sheet (last 5 examples from dataset.jsonl)
+
Current Task (UI tree + goal)
→ JSON action
```

**Technology:** llama-cpp-python with Qwen 2.5 (0.5B-2B params)

### 3. Action Layer (`actions/`)

**ActionController** (`actions/controller.py`)
- Executes JSON actions from Brain
- Supports: click, type, scroll, sys.launch, key.press
- Handles both "action" and "tool" key formats
- Maps friendly app names to executables

**Action Flow:**
```
Brain output → Controller.execute() → pyautogui/subprocess → System
```

### 4. Configuration (`config/`)

**ConfigManager** (`config/manager.py`)
- Loads `config.yaml` for settings
- Loads `.env` for secrets
- Provides dot-notation access: `config.get('model.n_ctx')`
- Environment variable overrides

### 5. Logging (`core/logging_utils.py`)

**Features:**
- Session-based log files with timestamps
- Automatic rotation (10MB max, 5 backups)
- JSON formatting option for parsing
- Multiple log levels (DEBUG, INFO, WARNING, ERROR)

## Data Flow

### Typical Execution Flow

```
1. User Input: "Open calculator and add 5 and 7"
   ↓
2. Main Loop initializes:
   - UIReader (eyes)
   - GhostEngine (brain)
   - ActionController (hands)
   ↓
3. Step 1:
   Eyes.capture_tree() → "Window: Desktop\n[1] Button: 'Start'..."
   ↓
   Brain.think(tree, goal) → {"action": "sys.launch", "app": "calculator"}
   ↓
   Hands.execute(action) → subprocess.Popen("calc.exe")
   ↓
4. Step 2:
   Eyes.capture_tree() → "Window: Calculator\n[24] Button: 'Five'..."
   ↓
   Brain.think(tree, goal) → {"action": "click", "id": 24}
   ↓
   Hands.execute(action) → UIReader.click_id(24)
   ↓
5. ... continues until {"action": "done"}
```

### Safety Mechanisms

```
┌─────────────────────────────┐
│   Main Loop Safety Layer    │
├─────────────────────────────┤
│ 1. Action History Tracking  │
│    • Last 5 actions stored  │
│    • Pattern detection      │
├─────────────────────────────┤
│ 2. Loop Detection           │
│    • Same action 3x = loop  │
│    • Auto-scroll fallback   │
├─────────────────────────────┤
│ 3. Screen Change Verify     │
│    • Compare UI tree        │
│    • Track stuck count      │
├─────────────────────────────┤
│ 4. Max Steps Limit          │
│    • Configurable via CLI   │
│    • Default: 20 steps      │
└─────────────────────────────┘
```

## Training Pipeline (Gym)

```
gym.py
  │
  ├─ 1. Launch safe app (Calculator, Notepad, etc.)
  │
  ├─ 2. Capture UI tree
  │
  ├─ 3. Filter safe clickable elements
  │      (Buttons, MenuItems, no "delete"/"format" keywords)
  │
  ├─ 4. Pick random element
  │
  ├─ 5. Click it
  │
  ├─ 6. Wait for screen change
  │
  ├─ 7. Verify screen changed?
  │     YES → Generate synthetic instruction → Save to dataset.jsonl
  │     NO  → Skip (no learning from failed clicks)
  │
  └─ 8. Loop (max 500 steps)
```

**Output:** `dataset.jsonl` with examples like:
```json
{
  "ui": "Window: Calculator\n[24] Button: 'Five'...",
  "goal": "Click number 5",
  "output": {"action": "click", "id": 24, "reason": "User wants to input 5"}
}
```

## Memory System: "Cheat Sheet"

Ghost-1 uses **few-shot learning** instead of fine-tuning:

1. Load last 5 examples from `dataset.jsonl`
2. Inject into prompt as "CHEAT SHEET"
3. LLM learns patterns instantly (no training required!)

**Benefits:**
- No GPU needed
- Instant updates (just add to dataset.jsonl)
- Small context usage (~1K tokens for 5 examples)

## Technology Stack

| Component | Technology | Purpose |
|-----------|----------|---------|
| UI Reading | pywinauto (UIA) | Extract UI element tree |
| LLM Inference | llama-cpp-python | Run local models (CPU) |
| Model | Qwen 2.5 (0.5B-2B) | Decision making |
| Actions | pyautogui, subprocess | Execute clicks, typing, launches |
| Config | PyYAML, python-dotenv | Settings management |
| Logging | Python logging + rotation | Session tracking |

## File Structure

```
Ghost-1/
├── main.py              # Entry point
├── gym.py               # Autonomous training
├── config.yaml          # Configuration
├── .env.example         # Secrets template
├── dataset.jsonl        # Training data
├── requirements.txt     # Dependencies
│
├── actions/
│   └── controller.py    # Action execution (Hands)
│
├── core/
│   ├── engine.py        # LLM inference (Brain)
│   ├── logging_utils.py # Logging system
│   └── prompts.py       # System prompts
│
├── vision/
│   ├── structure.py     # UI tree reader (Eyes)
│   ├── fast_vision.py   # Moondream integration
│   └── grid.py          # Grid overlay
│
├── config/
│   └── manager.py       # Config loader
│
├── models/              # .gguf model files
└── logs/                # Session logs
```

## Design Principles

### 1. Local-First
- No API calls
- No internet required
- Privacy-preserving

### 2. Small Models
- 0.5B-2B parameters
- CPU inference (<1s per decision)
- Low memory (<4GB RAM)

### 3. Structure Over Vision
- Parse UI accessibility tree (fast, reliable)
- Fallback to vision (Moondream) if needed
- No OCR for primary flow

### 4. Safety-First
- Whitelist safe apps in gym mode
- Filter dangerous keywords
- Loop detection and recovery
- Max steps limit

### 5. Teachable
- Teacher mode for human demonstrations
- Instant learning via few-shot examples
- No fine-tuning required

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Decision Time | 0.5-2s per step |
| Memory Usage | 2-4GB RAM |
| Model Size | 500MB-2GB (GGUF) |
| UI Scan Time | <100ms |
| Action Execution | <200ms |
| Context Window | 4096 tokens |

## Extension Points

### Adding New Actions

1. **Controller** (`actions/controller.py`):
   ```python
   elif act_type == "my.newaction":
       # implementation
       return "Success"
   ```

2. **Prompt** (`core/engine.py`):
   ```python
   SYSTEM_PROMPT_WITH_TOOLS += """
   7. MY NEW ACTION:
      {"action": "my.newaction", "param": "value"}
   """
   ```

### Adding New Safe Apps (Gym)

Edit `gym.py`:
```python
SAFE_APPS = ["Notepad", "Calculator", "MyNewApp"]
```

### Custom Configuration

Add to `config.yaml`:
```yaml
custom:
  setting1: value1
  setting2: value2
```

Access via:
```python
from config.manager import get_config
config = get_config()
value = config.get('custom.setting1')
```

## Future Architecture Considerations

See `todo.md` for planned enhancements:
- Multi-window handling
- Browser automation (Selenium/Playwright)
- Visual understanding (Moondream integration)
- State persistence across sessions
- Performance metrics dashboard
