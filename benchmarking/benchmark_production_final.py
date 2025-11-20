import requests
import json
import time
import numpy as np
from datetime import datetime

def log(m): print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)

def benchmark_production_api():
    """FINAL PRODUCTION BENCHMARK"""
    base_url = "http://localhost:8002"
    
    # Comprehensive test suite
    test_scenes = [
        # Urban scenes
        "a car near a building",
        "a person walking a dog in a park", 
        "a city street with cars and buildings",
        "a modern building with glass windows",
        
        # Natural scenes
        "a beautiful sunset over majestic mountains",
        "a tree beside a house with flowers",
        "a bird flying over water near mountains",
        "a peaceful lake surrounded by trees",
        "a mountain landscape with trees and water",
        
        # Mixed scenes
        "a person sitting on a bench in a garden",
        "a dog playing in a park with trees",
        "a car parked near a house with a garden",
        "a bird on a tree near a building"
    ]
    
    log("🚀 RUNNING FINAL PRODUCTION BENCHMARK...")
    
    results = []
    
    for scene in test_scenes:
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/describe/scene",
                json={
                    "scene_description": scene,
                    "enhance_adjectives": True,
                    "include_spatial": True,
                    "adjective_density": 1.0
                },
                timeout=10
            )
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                metrics = result["metrics"]
                
                results.append({
                    "input": scene,
                    "adjective_density": metrics["adjective_density"],
                    "adjective_count": metrics["adjective_count"],
                    "spatial_relations": metrics["spatial_relations"],
                    "processing_time": processing_time,
                    "output": result["enhanced_description"]
                })
                
                density_status = "✅" if metrics["adjective_density"] >= 0.3 else "⚠️"
                log(f"{density_status} {scene[:45]}... -> Density: {metrics['adjective_density']:.2f}")
            else:
                log(f"❌ Failed: {scene}")
                
        except Exception as e:
            log(f"❌ Error: {e}")
    
    # Calculate comprehensive metrics
    if results:
        avg_density = np.mean([r["adjective_density"] for r in results])
        avg_adjectives = np.mean([r["adjective_count"] for r in results])
        avg_spatial = np.mean([r["spatial_relations"] for r in results])
        avg_time = np.mean([r["processing_time"] for r in results])
        
        # Calculate success rate (density >= 0.3)
        success_rate = sum(1 for r in results if r["adjective_density"] >= 0.3) / len(results)
        
        print(f"\n" + "="*70)
        print("🎯 FINAL PRODUCTION BENCHMARK RESULTS")
        print("="*70)
        print(f"📊 PERFORMANCE METRICS:")
        print(f"   • Average Adjective Density: {avg_density:.3f}")
        print(f"   • Average Adjectives/Scene: {avg_adjectives:.1f}")
        print(f"   • Average Spatial Relations: {avg_spatial:.1f}")
        print(f"   • Average Processing Time: {avg_time*1000:.1f}ms")
        print(f"   • Success Rate (≥0.3 density): {success_rate:.1%}")
        print(f"   • Improvement vs Original: {((avg_density - 0.21) / 0.21 * 100):+.1f}%")
        
        print(f"\n🏆 COMPARISON TO PHASE 10 TARGETS:")
        print(f"   • Current Density: {avg_density:.3f} / Target: 4.000")
        print(f"   • Progress: {(avg_density / 4.0 * 100):.1f}% of target")
        
        print(f"\n🎯 SAMPLE OUTPUTS:")
        for i, result in enumerate(results[:4]):
            print(f"   {i+1}. Input: {result['input']}")
            print(f"      Output: {result['output']}")
            print(f"      Density: {result['adjective_density']:.2f}")
        
        print(f"\n📈 BENCHMARK STATUS: {'✅ SUCCESS' if avg_density >= 0.4 else '⚠️ NEEDS IMPROVEMENT'}")
        print("="*70)
        
        return {
            "avg_adjective_density": avg_density,
            "avg_adjectives_per_scene": avg_adjectives,
            "success_rate": success_rate,
            "improvement_percent": ((avg_density - 0.21) / 0.21 * 100),
            "progress_to_target": (avg_density / 4.0 * 100)
        }
    
    return None

if __name__ == "__main__":
    results = benchmark_production_api()
    if results and results["avg_adjective_density"] >= 0.4:
        print("\n🎉 EXCELLENT! Ready for article benchmarks!")
    elif results:
        print(f"\n⚠️  Good progress ({results['progress_to_target']:.1f}% of target), but needs more work")
