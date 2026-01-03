from datasets import load_dataset
print("Loading demo dataset...")
ds = load_dataset("lhoestq/demo1", split="train")
print(f"Loaded {len(ds)} rows.")
print(ds[0])
