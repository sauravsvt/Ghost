import sys
import os
import logging
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging to stdout
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

try:
    from vision.fast_vision import FastVision
    print("Attempting to load FastVision...")
    vision = FastVision(use_gpu=False)
    vision.load()
    
    if vision.model:
        print("SUCCESS: FastVision loaded.")
    else:
        print("FAILURE: FastVision did not load (check logs above).")

except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Exception: {e}")
