# Ghost

A Windows UI-automation agent. It reads the focused window, asks a local model for one action, and runs that action. The loop stops at `--max-steps` (default 20).

This is not a file-access runtime. It does not mediate paths, grants, or an audit log.

The model is a local GGUF file, loaded with `llama-cpp-python`. `core/engine.py` looks in `models/` for a Qwen GGUF, then any other `.gguf`. Pass `--model` to name a file.

## Run

On Windows, from this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

`--teacher` records the goal and the action you type into `dataset.jsonl`.

`python smart_gym.py` runs three cycles on the focused window. It skips Code and Terminal. When the window tree changes, it appends that action to `dataset_universal.jsonl`.
