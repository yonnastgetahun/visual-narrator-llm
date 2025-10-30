import time
from datetime import datetime

def monitor_fixed_training():
    print("🎯 MONITORING FIXED PHASE 2 TRAINING")
    print("=" * 50)
    print("✅ SUCCESS INDICATORS to watch for:")
    print("   1. Initial loss: 4.0-6.0 (NOT 0.0)")
    print("   2. Loss decreasing steadily")
    print("   3. Training time: 5-15 minutes")
    print("   4. Final model generates coherent text")
    print("")
    print("🚫 FAILURE INDICATORS (what we saw before):")
    print("   1. Loss stuck at 0.0")
    print("   2. Training takes hours")
    print("   3. Model outputs gibberish")
    print("")
    print("⏰ Started monitoring at:", datetime.now().strftime("%H:%M:%S"))
    print("💡 Run: python training/scale_up_phase_fixed.py")

if __name__ == "__main__":
    monitor_fixed_training()
