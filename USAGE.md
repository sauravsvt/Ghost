# Ghost-1 Usage Guide

## Quick Start

### Installation

```powershell
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download a model (if not already present)
# Place your .gguf model in the models/ directory
```

### Basic Usage

```powershell
# Run in AI mode
python main.py

# Run in Teacher mode (for data collection)
python main.py --teacher

# Run with custom settings
python main.py --max-steps 30 --verbose
```

## Command-Line Options

| Flag | Description | Default |
|------|-------------|---------|
| `--teacher` | Run in Teacher Mode for data collection | False |
| `--model PATH` | Path to .gguf model file | Auto-detect |
| `--verbose` | Enable DEBUG logging | False |
| `--log-level LEVEL` | Set log level (DEBUG/INFO/WARNING/ERROR) | INFO |
| `--json-logs` | Enable JSON formatted logs | False |
| `--max-steps N` | Maximum steps per task | 20 |
| `--version` | Show version and exit | - |

## Features

### 🎯 Available Actions

The Ghost-1 agent can perform these actions:

1. **Click** - Click any UI element by ID
   ```json
   {"action": "click", "id": 12}
   ```

2. **Type** - Type text into input fields
   ```json
   {"action": "type", "id": 5, "text": "hello world"}
   ```

3. **Scroll** - Scroll up or down in windows
   ```json
   {"action": "scroll.down"}
   {"action": "scroll", "direction": "up", "amount": 5}
   ```

4. **Launch Apps** - Open applications
   ```json
   {"action": "sys.launch", "app": "calculator"}
   ```
   
   Supported apps: `calculator`, `notepad`, `paint`, `chrome`, `edge`, `explorer`

5. **Wait** - Wait for UI to update
   ```json
   {"action": "wait"}
   ```

6. **Done** - Mark task as complete
   ```json
   {"action": "done"}
   ```

### 🛡️ Safety Features

- **Infinite Loop Detection**: Stops if same action repeats 3+ times
- **Screen Change Verification**: Tracks if actions have effect
- **Stuck Detection**: Auto-stops if no progress for 3 steps
- **Automatic Recovery**: Tries scrolling when stuck
- **Max Steps Limit**: Prevents runaway execution

### 📊 Logging

Ghost-1 uses session-based logging with automatic rotation:

- **Location**: `logs/ghost_YYYYMMDD_HHMMSS.log`
- **Rotation**: 10MB per file, keeps 5 backups
- **Formats**: Plain text (default) or JSON (`--json-logs`)
- **Levels**: DEBUG, INFO, WARNING, ERROR

Example with verbose logging:
```powershell
python main.py --verbose
# or
python main.py --log-level DEBUG
```

### 🎓 Training Mode (Gym)

Train the agent autonomously on safe applications:

```powershell
python gym.py
```

**Features:**
- Auto-explores Notepad, Calculator, Chrome, etc.
- Generates training data in `dataset.jsonl`
- Auto-recovery from minimized windows
- Safe element filtering

**Tips:**
1. Open a safe app (Notepad/Calculator) before running
2. Focus the app within 3 seconds of starting
3. Let it run for several minutes to generate data
4. Press Ctrl+C to stop anytime

### 📁 Configuration

Settings are managed through `config.yaml`:

```yaml
# Example configuration
model:
  path: "models/qwen1_5-1_8b-chat-q4_k_m.gguf"
  n_ctx: 4096
  
agent:
  max_steps: 20
  action_delay: 0.5
```

For secrets, use `.env` file (copy from `.env.example`):
```bash
MODEL_PATH=/path/to/model.gguf
```

## Examples

### Example 1: Simple Calculator Task

```powershell
python main.py
```

User Input: `Open calculator and add 5 and 7`

Expected flow:
1. Launches Calculator
2. Clicks "5"
3. Clicks "+"
4. Clicks "7"
5. Clicks "="
6. Done!

### Example 2: Notepad Editing

```powershell
python main.py --max-steps 15
```

User Input: `Open notepad and type hello world`

Expected flow:
1. Launches Notepad
2. Types "hello world"
3. Done!

### Example 3: Data Collection

```powershell
python main.py --teacher
```

Teach the agent by demonstrating actions:
```
User Goal: Click the File menu
> click 2

User Goal: Open a new file
> click 5
```

All demonstrations are saved to `dataset.jsonl`.

## Troubleshooting

### "No valid window focused"
**Problem**: Gym can't find a safe app to interact with.

**Solution**: 
1. Open Calculator or Notepad
2. Click inside the app to give it focus
3. Run `python gym.py`
4. Within 3 seconds, click back into the app

### "Model not found"
**Problem**: Can't locate .gguf model file.

**Solution**:
1. Download a Qwen 2.5 model
2. Place in `models/` directory
3. Or specify path: `python main.py --model path/to/model.gguf`

### "Infinite loop detected"
**Problem**: Agent keeps repeating same action.

**Solution**: This is normal! The agent auto-recovers by:
1. Detecting the loop after 3 repetitions
2. Trying to scroll to find new elements
3. Stopping if still stuck after 3 attempts

### Import Errors
**Problem**: Missing dependencies.

**Solution**:
```powershell
# Activate venv
.\.venv\Scripts\Activate.ps1

# Install all dependencies
pip install -r requirements.txt
```

## Advanced Usage

### Custom Action Delay

Slow down actions for debugging:

```yaml
# config.yaml
agent:
  action_delay: 2.0  # 2 seconds between actions
```

### JSON Logs for Analysis

```powershell
python main.py --json-logs
```

Parse logs programmatically:
```python
import json

with open('logs/ghost_20260103_120000.log') as f:
    for line in f:
        log = json.loads(line)
        print(f"{log['level']}: {log['message']}")
```

### Integration with External Tools

Use Ghost-1 programmatically:

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

## Contributing

See `todo.md` for planned features and improvements.

To add a new action:
1. Add handler in `actions/controller.py`
2. Update `SYSTEM_PROMPT_WITH_TOOLS` in `core/engine.py`
3. Add tests
4. Update this documentation

## License

See the main repository for license information.
