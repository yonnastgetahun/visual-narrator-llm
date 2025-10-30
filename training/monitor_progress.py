import time
import os
from datetime import datetime

def monitor_training():
    log_file = "./logs/phase2/training_log.json"
    output_dir = "./outputs/phase2_run"
    
    print("🔍 Training Progress Monitor")
    print("⏰ Started at:", datetime.now().strftime("%H:%M:%S"))
    print("📊 Monitoring logs in:", log_file)
    print("💾 Output will be saved to:", output_dir)
    print("\nExpected timeline:")
    print("├── 0-2 min: Model loading and data preparation")
    print("├── 2-8 min: Training (you'll see loss decreasing)")
    print("└── 8-10 min: Model saving and completion")
    print("\n💡 You can check progress in the logs/phase2/ folder")
    print("🎯 Look for 'loss' values going down over time!")

if __name__ == "__main__":
    monitor_training()
