import os
import requests
import subprocess
import sys

# Configuration
MODEL_URL = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "qwen2.5-0.5b-instruct-q4_k_m.gguf")

def install_dependencies():
    print("[*] Installing Windows Automation dependencies...")
    packages = ["pywinauto", "llama-cpp-python", "colorama", "requests"]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)

def download_model():
    if os.path.exists(MODEL_PATH):
        print(f"[+] Model already exists at {MODEL_PATH}")
        return

    print(f"[*] Downloading Qwen 2.5 0.5B (Tiny Giant)...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    response = requests.get(MODEL_URL, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(MODEL_PATH, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            
    print("[+] Model Downloaded Successfully.")

if __name__ == "__main__":
    try:
        install_dependencies()
        download_model()
        print("\n[✓] Setup Complete. You are ready for Teacher Mode.")
        print("Run: python main.py --teacher")
    except Exception as e:
        print(f"[!] Setup Failed: {e}")
