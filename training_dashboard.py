import time
import subprocess
import json
import glob
import os
from datetime import datetime

def get_gpu_status():
    """Get GPU utilization and memory"""
    try:
        result = subprocess.run([
            "nvidia-smi", 
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split(',')
            return {
                "utilization": f"{gpu_info[0]}%",
                "memory": f"{gpu_info[1]}/{gpu_info[2]} MB",
                "temperature": f"{gpu_info[3]}°C"
            }
    except:
        pass
    return {"utilization": "N/A", "memory": "N/A", "temperature": "N/A"}

def get_training_status():
    """Check if training is running"""
    try:
        result = subprocess.run(["pgrep", "-f", "train_synth_blip.py"], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def get_latest_log():
    """Get latest training log content"""
    log_files = glob.glob("logs/train_enhanced_*.log")
    if log_files:
        latest_log = sorted(log_files)[-1]
        with open(latest_log, 'r') as f:
            lines = f.readlines()[-10:]  # Last 10 lines
        return "".join(lines)
    return "No log file found"

def get_checkpoint_count():
    """Count checkpoints in output directory"""
    checkpoints = glob.glob("outputs/phase7_enhanced_training/checkpoint-*")
    return len(checkpoints)

def training_dashboard():
    """Real-time training dashboard"""
    
    print("🎯 Visual Narrator VLM - Training Dashboard")
    print("=" * 60)
    
    while True:
        try:
            # Clear screen (optional)
            os.system('clear')
            
            print("🎯 Visual Narrator VLM - Training Dashboard")
            print("=" * 60)
            print(f"🕐 Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # Training status
            is_training = get_training_status()
            status_emoji = "🟢" if is_training else "🔴"
            print(f"{status_emoji} Training Status: {'RUNNING' if is_training else 'STOPPED'}")
            
            # GPU status
            gpu_status = get_gpu_status()
            print(f"🖥️  GPU Utilization: {gpu_status['utilization']}")
            print(f"💾 GPU Memory: {gpu_status['memory']}")
            print(f"🌡️  GPU Temperature: {gpu_status['temperature']}")
            
            # Checkpoints
            checkpoint_count = get_checkpoint_count()
            print(f"💾 Checkpoints: {checkpoint_count}")
            
            # Latest log
            print(f"📋 Latest Log Snippet:")
            print("-" * 40)
            log_content = get_latest_log()
            print(log_content)
            print("-" * 40)
            
            print("\nPress Ctrl+C to exit dashboard")
            time.sleep(10)  # Update every 10 seconds
            
        except KeyboardInterrupt:
            print("\n🛑 Dashboard stopped")
            break

if __name__ == "__main__":
    training_dashboard()
