import os
import json
from datetime import datetime

def log(m): print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)

def validate_assets():
    """Validate that all our trained assets exist and are accessible"""
    log("🔍 VALIDATING TRAINED ASSETS...")
    
    assets_to_check = [
        ("phase9/spatial_predictor_model.pth", "Trained Spatial Predictor"),
        ("outputs/learned_spatial_patterns.json", "Phase 8 Spatial Patterns"),
        ("phase8/comprehensive_spatial_dataset.json", "Comprehensive Spatial Dataset"),
        ("phase8/pattern_generated_spatial.json", "Pattern Generated Dataset"),
        ("phase9/multi_object_scenes.json", "Multi-Object Scenes Dataset")
    ]
    
    existing_assets = []
    missing_assets = []
    
    for path, description in assets_to_check:
        if os.path.exists(path):
            existing_assets.append((path, description))
            
            # Get some stats for existing assets
            if path.endswith('.json'):
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    log(f"   ✅ {description}: {len(data)} items")
                except:
                    log(f"   ✅ {description}: EXISTS (error reading)")
            else:
                size_mb = os.path.getsize(path) / (1024 * 1024)
                log(f"   ✅ {description}: {size_mb:.1f} MB")
                
        else:
            missing_assets.append((path, description))
            log(f"   ❌ {description}: MISSING")
    
    log(f"\n📊 ASSET VALIDATION SUMMARY:")
    log(f"   ✅ Existing: {len(existing_assets)} assets")
    log(f"   ❌ Missing: {len(missing_assets)} assets")
    
    if missing_assets:
        log("\n⚠️  MISSING ASSETS:")
        for path, description in missing_assets:
            log(f"   • {description}: {path}")
    
    return len(existing_assets) > 0

if __name__ == "__main__":
    success = validate_assets()
    if success:
        log("\n🎯 ASSETS VALIDATED - READY FOR BENCHMARKING!")
    else:
        log("\n❌ CRITICAL ASSETS MISSING - CHECK PATHS!")
