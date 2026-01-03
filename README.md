# Ghost-1: Terminal Agent

> **Local-First | Structure-Based UI | Qwen 2.5 0.5B-2B**

A fully local, privacy-first autonomous desktop agent that runs on CPU. No cloud, no API calls, no data leaves your machine.

```
╔════════════════════════════════════════════╗
║    GHOST-1: LOCAL STRUCTURE (v5.0)         ║
║   Qwen 2.5 | pywinauto | Cheat Sheet AI   ║
╚════════════════════════════════════════════╝
```

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

## 🚀 Quick Start

```powershell
# Activate virtual environment (Windows)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run Ghost-1
python main.py

# Run autonomous training
python gym.py
```

## ✨ Features

### 🎯 Capabilities
- ✅ **Click** any UI element by ID (blind clicking)
- ✅ **Type** text into fields  
- ✅ **Scroll** up/down in windows
- ✅ **Launch** apps (Calculator, Notepad, Chrome, etc.)
- ✅ **Learn** from demonstrations (Teacher mode)
- ✅ **Train** autonomously (Gym mode)

### 🛡️ Safety & Intelligence
- **Infinite Loop Detection** - Stops if same action repeats 3+ times
- **Screen Change Verification** - Ensures actions have effect
- **Stuck Detection** - Auto-stops if no progress for 3 steps  
- **Automatic Recovery** - Tries scrolling when stuck
- **Configurable Limits** - Max steps, timeouts, etc.

### 📊 Logging & Config
- **Session-Based Logs** - Timestamped log files per run
- **JSON Logging** - Optional structured logs for parsing
- **Log Rotation** - 10MB max, 5 backups
- **YAML Config** - Centralized settings in `config.yaml`
- **Environment Variables** - Secrets in `.env` file

## 📚 Documentation

- **[USAGE.md](USAGE.md)** - Complete usage guide with examples and troubleshooting
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design and technical details
- **[todo.md](todo.md)** - Planned features and improvements

## 🎓 Training Modes

### AI Mode (Default)
Let the agent decide actions autonomously:
```powershell
python main.py
```

### Teacher Mode
Demonstrate actions for training data:
```powershell
python main.py --teacher
```

### Gym Mode
Autonomous exploration and data generation:
```powershell
python gym.py
```

## 🔧 CLI Options

| Flag | Description |
|------|-------------|
| `--verbose` | Enable DEBUG logging |
| `--log-level LEVEL` | Set log level (DEBUG/INFO/WARNING/ERROR) |
| `--json-logs` | Enable JSON formatted logs |
| `--max-steps N` | Maximum steps per task |
| `--version` | Show version |
| `--teacher` | Run in Teacher Mode |
| `--model PATH` | Path to .gguf model |

## 🧠 How It Works

### 1. Structure-Based UI Reading
Uses **pywinauto** with UIA (UI Automation) backend to extract accessibility tree:
```
Window: Calculator
[24] Button: 'Five'
[25] Button: 'Six'
[26] Button: 'Plus'
```

### 2. Local LLM Decision Making
**Qwen 2.5** (0.5B-2B params) via llama-cpp-python:
- Processes UI tree + user goal
- Outputs JSON action
- Uses "Cheat Sheet" (last 5 examples from dataset.jsonl)
- CPU-only inference (<2s per decision)

### 3. Action Execution
Maps LLM output to system actions:
```json
{"action": "click", "id": 24}
→ UIReader.click_id(24)
→ pywinauto clicks element
```

### 4. Safety Loop
- Tracks action history
- Detects loops (same action 3x)
- Verifies screen changes
- Auto-recovers with scroll

## 📁 Project Structure

```
Ghost-1/
├── main.py              # Entry point
├── gym.py               # Autonomous training
├── config.yaml          # Configuration
├── dataset.jsonl        # Training data
│
├── core/
│   ├── engine.py        # LLM inference
│   └── logging_utils.py # Logging system
│
├── vision/
│   └── structure.py     # UI tree reader
│
├── actions/
│   └── controller.py    # Action execution
│
├── config/
│   └── manager.py       # Config loader
│
└── logs/                # Session logs
```

## 🔒 Privacy Guarantee

- ✅ 100% local execution
- ✅ No API calls
- ✅ No telemetry
- ✅ All data on your machine
- ✅ GDPR compliant

## 📊 Performance

| Metric | Value |
|--------|-------|
| RAM Usage | 2-4GB |
| Decision Time | 0.5-2s |
| Model Size | 500MB-2GB (GGUF) |
| Platform | Windows (CPU-only) |

## 🚧 Recent Improvements

- ✅ **Scroll Support** - Navigate long pages
- ✅ **Loop Detection** - No more infinite loops
- ✅ **Screen Verification** - Ensures progress
- ✅ **Better Logging** - Session tracking & rotation
- ✅ **Config System** - YAML + .env management
- ✅ **CLI Flags** - Verbose, log levels, etc.

## 📜 License

COPYRIGHT (c) 2025 - SAURAV SHRIWASTAV

---

**Ghost-1 v5.0** - Structure/Local Architecture
