try:
    from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer
    print("✅ Transformers: WORKING")
except Exception as e:
    print(f"❌ Transformers: {e}")

try:
    import accelerate
    print(f"✅ Accelerate: {accelerate.__version__}")
except Exception as e:
    print(f"❌ Accelerate: {e}")

try:
    import datasets
    print("✅ Datasets: WORKING")
except Exception as e:
    print(f"❌ Datasets: {e}")
