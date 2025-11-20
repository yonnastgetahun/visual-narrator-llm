import requests
import json
import time
import numpy as np
from datetime import datetime

def log(m): print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)

def benchmark_fixed_api():
    """Benchmark the fixed high density API"""
    base_url = "http://localhost:8001"
    
    test_scenes = [
        "a car near a building",
        "a person walking a dog in a park", 
        "a beautiful sunset over majestic mountains",
        "a tree beside a house with flowers",
        "a bird flying over water near mountains",
        "a city street with cars and buildings",
        "a peaceful lake surrounded by trees",
        "a modern building with glass windows",
        "a mountain landscape with trees and water",
        "a person sitting on a bench in a garden"
    ]
    
    log("🚀 BENCHMARKING FIXED HIGH DENSITY API...")
    
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
                
                log(f"✅ {scene[:40]}... -> Density: {metrics['adjective_density']:.2f}")
            else:
                log(f"❌ Failed: {scene}")
                
        except Exception as e:
            log(f"❌ Error: {e}")
    
    # Calculate averages
    if results:
        avg_density = np.mean([r["adjective_density"] for r in results])
        avg_adjectives = np.mean([r["adjective_count"] for r in results])
        avg_spatial = np.mean([r["spatial_relations"] for r in results])
        avg_time = np.mean([r["processing_time"] for r in results])
        
        print(f"\n📊 FIXED HIGH DENSITY BENCHMARK RESULTS:")
        print(f"   • Average Adjective Density: {avg_density:.3f}")
        print(f"   • Average Adjectives/Scene: {avg_adjectives:.1f}")
        print(f"   • Average Spatial Relations: {avg_spatial:.1f}")
        print(f"   • Average Processing Time: {avg_time*1000:.1f}ms")
        print(f"   • Improvement vs Original: {((avg_density - 0.21) / 0.21 * 100):+.1f}%")
        
        print(f"\n🎯 SAMPLE OUTPUTS:")
        for i, result in enumerate(results[:3]):
            print(f"   {i+1}. Input: {result['input']}")
            print(f"      Output: {result['output']}")
            print(f"      Density: {result['adjective_density']:.2f}")
        
        return {
            "avg_adjective_density": avg_density,
            "avg_adjectives_per_scene": avg_adjectives,
            "improvement_percent": ((avg_density - 0.21) / 0.21 * 100)
        }
    
    return None

if __name__ == "__main__":
    benchmark_fixed_api()
