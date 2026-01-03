# Ghost-1: Autonomous Desktop Agent v6.1

> **Local-First | Dialog Intelligence | Causal Learning**  
> Privacy-preserving autonomous agent that runs entirely on CPU with 0.5B-2B parameter models

```
╔════════════════════════════════════════════╗
║    GHOST-1 v6.1: SMART GYM EDITION         ║
║   Qwen 2.5 | pywinauto | Dialog Handling  ║
╚════════════════════════════════════════════╝
```

## 🚀 Quick Start

```powershell
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Ghost-1
python main.py

# 4. Run autonomous training (NEW!)
python smart_gym.py
```

## ✨ What's New in v6.1

### 🧠 Smart Gym v2.1 - True Autonomy
- **Dialog Intelligence**: Automatically handles sign-in popups, file pickers, alerts
- **Context Detection**: Distinguishes dialogs from true window failures
- **Causal Learning**: Learns from mistakes (hazards vs neutral dialogs)
- **Pain Sensors**: Detects when actions break apps (minimize, close)
- **3-Way Classification**: Positive, Dialog (neutral), Hazard (avoid forever)

### 🛡️ Core Improvements
- **Scroll Capability**: scroll_down/scroll_up/scroll actions
- **Screen Verification**: Detects infinite loops, stuck states
- **Configuration Management**: config.yaml + .env support
- **Structured Logging**: Session-based logs with rotation
- **CLI Enhancements**: --verbose, --log-level, --max-steps, --version

### 🐛 Stability Fixes
- Hallucination patches (action name validation)
- Type validation for IDs (graceful errors)
- Context window doubled to 8192
- Aggressive UI filtering (60 elements max)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         GHOST-1                             │
├─────────────────┬─────────────────┬─────────────────────────┤
│      BRAIN      │      EYES       │         HANDS           │
│   (core/)       │   (vision/)     │       (actions/)        │
├─────────────────┼─────────────────┼─────────────────────────┤
│ • Qwen 2.5      │ • UI Automation │ • PyAutoGUI Actions     │
│   0.5B-2B       │   Tree (UIA)    │ • App Launching         │
│ • llama-cpp     │ • pywinauto     │ • Scroll Support        │
│ • Few-Shot      │ • ID-Based      │ • Loop Detection        │
│   Learning      │   Clicking      │ • Auto-Recovery         │
└─────────────────┴─────────────────┴─────────────────────────┘
```

### Vision Layer (`vision/`)
**UIReader** - Scans active window, extracts UI tree, assigns IDs
- Technology: pywinauto (UIA backend)
- Speed: <100ms per scan
- Filtering: 60 elements max, skip containers

### Brain Layer (`core/`)
**GhostEngine** - Local LLM with few-shot learning
- Model: Qwen 2.5 (0.5B-2B params)
- Context: 8192 tokens
- Memory: Last 5 examples from dataset.jsonl ("Cheat Sheet")

### Action Layer (`actions/`)
**ActionController** - Executes JSON actions
- Supports: click, type, scroll, sys.launch, wait, done
- Technology: pyautogui + subprocess

---

## 📚 Usage Guide

### Command-Line Options

| Flag | Description | Default |
|------|-------------|---------|
| `--teacher` | Run in Teacher Mode for data collection | False |
| `--model PATH` | Path to .gguf model file | Auto-detect |
| `--verbose` | Enable DEBUG logging | False |
| `--log-level LEVEL` | Set log level (DEBUG/INFO/WARNING/ERROR) | INFO |
| `--json-logs` | Enable JSON formatted logs | False |
| `--max-steps N` | Maximum steps per task | 20 |
| `--version` | Show version and exit | - |

### Available Actions

1. **Click** - Click any UI element by ID
   ```json
   {"action": "click", "id": 12}
   ```

2. **Type** - Type text into input fields
   ```json
   {"action": "type", "id": 5, "text": "hello world"}
   ```

3. **Scroll** - Scroll in windows
   ```json
   {"action": "scroll.down"}
   {"action": "scroll", "direction": "up", "amount": 5}
   ```

4. **Launch Apps** - Open applications
   ```json
   {"action": "sys.launch", "app": "calculator"}
   ```
   Supported: `calculator`, `notepad`, `paint`, `chrome`, `edge`, `explorer`

5. **Wait** - Wait for UI to update
   ```json
   {"action": "wait"}
   ```

6. **Done** - Mark task complete
   ```json
   {"action": "done"}
   ```

### Safety Features

- **Loop Detection**: Stops if same action repeats 3+ times
- **Screen Change Verification**: Tracks if actions have effect
- **Stuck Detection**: Auto-stops if no progress for 3 steps
- **Auto-Recovery**: Tries scrolling when stuck
- **Max Steps Limit**: Prevents runaway execution
- **Dialog Dismissal**: Automatically handles popups with ESC

### Logging

- **Location**: `logs/ghost_YYYYMMDD_HHMMSS.log`
- **Rotation**: 10MB per file, keeps 5 backups
- **Formats**: Plain text or JSON (`--json-logs`)
- **Levels**: DEBUG, INFO, WARNING, ERROR

Example:
```powershell
python main.py --verbose --log-level DEBUG
```

---

## 🎓 Smart Gym Training

Train the agent autonomously with **Smart Gym v2.1**:

```powershell
python smart_gym.py
```

### Features

**🧠 Dialog Intelligence**
- Detects popups: "Sign in", "Save as", "Print", alerts
- Auto-dismisses with ESC + Cancel button strategies
- Continues training without interruption

**💡 Causal Learning**
- **Positive**: Action worked, screen changed → Save example
- **Dialog**: Popup appeared → Dismiss & continue (neutral)
- **Hazard**: Window disappeared → Blacklist forever & relaunch

**🎯 Training Circuit**
- Rotates through: Notepad, Paint, Calculator, WordPad
- 50 actions per app × 5 cycles = 250 training attempts
- Automatic app launch and cleanup

**📊 Output**
- Saves to `dataset.jsonl` with result classification
- Hazard memory prevents repeating destructive actions
- Dialog triggers tracked separately

### Example Output

```
[1/50] Trying: User avatar (ID 15)
[⚠] HAZARD: 'User avatar' killed the window!
[📝] Saved negative example. Will avoid ID 15

[2/50] Trying: Lists (ID 9)
[💬] DIALOG: 'Lists' opened a popup
[✓] Dialog dismissed with ESC

[3/50] Trying: Headings (ID 12)
[+] SUCCESS: Learned 'Headings' -> ID 12
```

### Tips

1. Let Smart Gym run for 15-30 minutes
2. It will auto-recover from minimized windows
3. Press Ctrl+C to stop anytime
4. Review `dataset.jsonl` for quality

---

## ⚙️ Configuration

### config.yaml

```yaml
# Example configuration
model:
  path: "models/qwen1_5-1_8b-chat-q4_k_m.gguf"
  n_ctx: 8192
  temperature: 0.2
  
agent:
  max_steps: 20
  action_delay: 0.5
```

### .env (secrets)

```bash
MODEL_PATH=/path/to/model.gguf
```

Copy from `.env.example` and customize.

---

## 📊 Examples

### Example 1: Calculator

```powershell
python main.py
```

**Input**: "Open calculator and add 5 and 7"

**Flow**:
1. Launches Calculator
2. Clicks "5"
3. Clicks "+"
4. Clicks "7"
5. Clicks "="
6. Done!

### Example 2: Notepad

```powershell
python main.py --max-steps 15
```

**Input**: "Open notepad and type hello world"

**Flow**:
1. Launches Notepad
2. Types "hello world"
3. Done!

### Example 3: Data Collection

```powershell
python main.py --teacher
```

**Teach the agent**:
```
User Goal: Click the File menu
> click 2

User Goal: Open a new file
> click 5
```

All saved to `dataset.jsonl`.

---

## 🛠️ Programmatic Usage

```python
from core.engine import GhostEngine
from vision.structure import UIReader
from actions.controller import ActionController

# Initialize
reader = UIReader()
brain = GhostEngine()
hands = ActionController(reader)

# Get UI state
ui_tree = reader.capture_tree()

# Think
action = brain.think(ui_tree, "Open calculator")

# Act
result = hands.execute(action)
```

---

## 🚨 Troubleshooting

### "No valid window focused"
**Solution**: 
1. Open Calculator or Notepad
2. Click inside the app
3. Run `python smart_gym.py`
4. Within 3 seconds, click back into the app

### "Model not found"
**Solution**:
1. Download a Qwen 2.5 model (.gguf)
2. Place in `models/` directory
3. Or specify: `python main.py --model path/to/model.gguf`

### "Infinite loop detected"
This is normal! The agent auto-recovers by:
1. Detecting the loop after 3 repetitions
2. Trying to scroll to find new elements
3. Stopping if still stuck

### Import Errors
```powershell
# Activate venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## 📁 File Structure

```
Ghost-1/
├── main.py                # Entry point
├── smart_gym.py           # Smart Gym v2.1 (NEW!)
├── mega_gym.py            # Original autonomous trainer
├── gym.py                 # Basic trainer
├── config.yaml            # Configuration
├── .env.example           # Secrets template
├── dataset.jsonl          # Training data
├── requirements.txt       # Dependencies
│
├── actions/
│   └── controller.py      # Action execution
│
├── core/
│   ├── engine.py          # LLM inference
│   ├── logging_utils.py   # Logging system
│   └── prompts.py         # System prompts
│
├── vision/
│   ├── structure.py       # UI tree reader
│   ├── fast_vision.py     # Moondream integration
│   └── grid.py            # Grid overlay
│
├── config/
│   └── manager.py         # Config loader
│
├── models/                # .gguf model files
└── logs/                  # Session logs
```

---

## 🎯 Design Principles

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
- Whitelist safe apps
- Filter dangerous keywords
- Loop detection and recovery
- Dialog intelligence

### 5. Teachable
- Teacher mode for demonstrations
- Few-shot learning (no fine-tuning!)
- Instant updates via dataset.jsonl

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| Decision Time | 0.5-2s per step |
| Memory Usage | 2-4GB RAM |
| Model Size | 500MB-2GB (GGUF) |
| UI Scan Time | <100ms |
| Action Execution | <200ms |
| Context Window | 8192 tokens |

---

## 🔧 Extension Points

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

### Adding Training Apps

Edit `smart_gym.py`:
```python
TRAINING_CIRCUIT = {
    "notepad.exe": "Notepad",
    "mynewapp.exe": "MyNewApp",
}
```

---

## 🎉 What Makes v6.1 Special?

**Before v6.1:**
- ❌ Gets stuck on sign-in dialogs
- ❌ Can't recover from minimize
- ❌ No distinction between failures and popups

**After v6.1:**
- ✅ Handles ANY dialog automatically
- ✅ Learns which buttons are dangerous
- ✅ Classifies outcomes with context awareness
- ✅ **True autonomy** - runs for hours unsupervised!

---

## 📝 Contributing

See `todo.md` for planned features.

To add a feature:
1. Update code in relevant layer (Brain/Eyes/Hands)
2. Update system prompt if needed
3. Add to configuration
4. Update this README

---

## 📄 License

See the main repository for license information.

---

**Ghost-1 v6.1** - Built with ❤️ for local-first AI automation
