# Ghost-1: Terminal Agent

> **Bi-Mamba 1.58-bit Brain | Spatial-Mamba Vision | CALM Decoding**

A fully local, privacy-first autonomous desktop agent that runs on CPU. No cloud, no API calls, no data leaves your machine.

```
╔════════════════════════════════════════════╗
║      GHOST-1: TERMINAL AGENT (v1.0)        ║
║   Bi-Mamba 1.58-bit | Spatial-Mamba Vision ║
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
│ • Bi-Mamba      │ • MSS Capture   │ • PyAutoGUI Controller  │
│   1.58-bit      │ • Spatial-Mamba │ • Human-like Movement   │
│ • CALM Decoder  │   Linearization │ • File/CMD Tools        │
│ • GRPO Trainer  │ • Grid System   │ • Safety Sandboxing     │
└─────────────────┴─────────────────┴─────────────────────────┘
```

## 🚀 Quick Start

```bash
# Clone/navigate to project
cd ghost-1-cli

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Ghost-1
python main.py
```

## 📁 Project Structure

```
ghost-1-cli/
│
├── core/                    # The Brain
│   ├── engine.py           # Bi-Mamba 1.58-bit inference wrapper
│   ├── calm_decoder.py     # Speculative multi-token decoding
│   └── trainer.py          # GRPO self-evolution loop
│
├── vision/                  # The Eyes
│   ├── perceptor.py        # MSS screen capture + preprocessing
│   └── grid.py             # Coordinate translation system
│
├── actions/                 # The Hands
│   ├── controller.py       # PyAutoGUI with human-like movement
│   └── tools.py            # File ops + command execution
│
├── config/                  # Configuration
│   ├── settings.py         # Constants and hyperparameters
│   └── prompts.py          # System prompts for reasoning
│
├── logs/                    # Runtime logs
├── main.py                  # CLI entry point
├── requirements.txt         # Dependencies
└── setup.py                 # Package installation
```

## 🧠 How It Works

### 1. Observe → Think → Act Loop

```
User Input
    ↓
┌─────────────────────────────────────┐
│  OBSERVE: Capture screen via MSS    │
│  ↓                                  │
│  THINK: Generate <think> trace      │
│  ↓                                  │
│  ACT: Execute JSON tool call        │
│  ↓                                  │
│  VERIFY: Check result, retry if     │
│          needed (self-correction)   │
└─────────────────────────────────────┘
    ↓
Loop until task complete
```

### 2. Reasoning Format

The model outputs DeepSeek-R1 style reasoning:

```xml
<think>
User wants to open Chrome and search for "Python tutorials".
1. Observation: Desktop visible. Chrome icon at (45, 890).
2. Plan: Click Chrome → Wait for load → Type in search bar.
3. Constraint: Ensure Chrome is not already maximized.
</think>
{
    "tool": "mouse.click",
    "x": 45,
    "y": 890
}
```

### 3. Available Tools

| Tool | Arguments | Description |
|------|-----------|-------------|
| `mouse.click` | `x`, `y` | Click at coordinates |
| `mouse.move` | `x`, `y` | Move mouse (no click) |
| `keyboard.type` | `text` | Type text naturally |
| `keyboard.hotkey` | `keys` | Press key combination |
| `browser.open` | `url` | Open URL in browser |
| `file.read` | `path` | Read file contents |
| `file.write` | `path`, `content` | Write to file |
| `os.command` | `command` | Execute shell command |
| `wait` | `seconds` | Pause execution |
| `done` | `message` | Signal task completion |

## 🛡️ Safety Features

1. **Failsafe**: Move mouse to top-left corner (0,0) to instantly stop the agent
2. **Path Sandboxing**: File operations restricted to workspace directory
3. **Command Blocklist**: Dangerous commands (`rm -rf`, `format`, etc.) are blocked
4. **Rate Limiting**: Maximum 60 actions per minute
5. **Confirmation Required**: Delete operations need explicit confirmation

## ⚙️ Configuration

Edit `config/settings.py` to customize:

```python
# Model settings
MODEL.temperature = 0.2        # Lower = more deterministic
MODEL.context_length = 1_000_000  # Mamba handles infinite context

# Safety settings
SAFETY.failsafe_enabled = True
SAFETY.max_actions_per_minute = 60

# Training (disabled by default)
TRAINING.enabled = False
TRAINING.learning_rate = 1e-5
```

## 🔧 Extending Ghost-1

### Adding New Tools

Edit `actions/tools.py`:

```python
class CustomTools:
    def my_new_tool(self, args: dict) -> ToolResult:
        # Your implementation
        return ToolResult(success=True, output="Done!")
```

Then register in `actions/controller.py`:

```python
elif tool_name == "my.new.tool":
    self.custom_tools.my_new_tool(args)
```

### Swapping in Real Model

Replace the simulation in `core/engine.py`:

```python
# From:
self.model_state = {"status": "loaded", "quantization": "ternary_1.58"}

# To:
from ghost_cpp import load_bimamba_quantized
self.model_state = load_bimamba_quantized(self.config.model_path)
```

## 📊 Performance

| Metric | Value |
|--------|-------|
| RAM Usage | ~1.2GB (model) + ~200MB (runtime) |
| Inference | CPU-only, optimized for AMD Ryzen |
| Latency | ~100ms per action (simulated) |
| Context | 1M tokens (Mamba linear complexity) |

## 🔒 Privacy Guarantee

- ✅ 100% local execution
- ✅ No API calls to external services
- ✅ No telemetry or analytics
- ✅ All data stays on your machine
- ✅ GDPR compliant by design

## 📜 License

MIT License - Use freely, modify as needed.

---

**Built for the "Late 2025" architecture spec: Bi-Mamba 1.58-bit + Spatial-Mamba + CALM**
