import torch
import time
import json
import pandas as pd
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import glob

def benchmark_speed_quality():
    """Benchmark trade-off between speed and quality"""
    
    print("⚡ SPEED vs QUALITY BENCHMARK")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_images = glob.glob("/data/coco/train2017/*.jpg")[:10]
    
    models_to_test = [
        ("Our-VLM-FP16", "outputs/phase7_3_large_scale/checkpoint-step-5000-1762322982", "fp16"),
        ("Our-VLM-FP32", "outputs/phase7_3_large_scale/checkpoint-step-5000-1762322982", "fp32"),
        ("BLIP-Base", "Salesforce/blip-image-captioning-base", "fp16"),
        ("BLIP-Large", "Salesforce/blip-image-captioning-large", "fp16")
    ]
    
    results = []
    
    for model_name, model_path, precision in models_to_test:
        print(f"\n🧪 Testing {model_name} ({precision})...")
        
        try:
            processor = BlipProcessor.from_pretrained(model_path)
            
            if "Our-VLM" in model_name and precision == "fp32":
                model = BlipForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=torch.float32
                ).to(device)
            else:
                model = BlipForConditionalGeneration.from_pretrained(model_path).to(device)
            
            # Warmup
            warmup_image = Image.open(test_images[0])
            warmup_inputs = processor(images=warmup_image, return_tensors="pt").to(device)
            _ = model.generate(**warmup_inputs, max_length=30)
            
            # Benchmark
            times = []
            adjectives = []
            
            for img_path in test_images:
                image = Image.open(img_path)
                inputs = processor(images=image, return_tensors="pt").to(device)
                
                start_time = time.time()
                
                if precision == "fp16":
                    with torch.amp.autocast("cuda", enabled=True):
                        outputs = model.generate(**inputs, max_length=50)
                else:
                    outputs = model.generate(**inputs, max_length=50)
                
                inference_time = time.time() - start_time
                caption = processor.decode(outputs[0], skip_special_tokens=True)
                
                # Simple adjective count
                adj_count = len([w for w in caption.split() if any(adj in w for adj in 
                                ['vivid', 'gleaming', 'rugged', 'tranquil', 'velvety', 'golden'])])
                
                times.append(inference_time)
                adjectives.append(adj_count)
            
            if times and adjectives:
                avg_time = sum(times) / len(times) * 1000  # Convert to ms
                avg_adjectives = sum(adjectives) / len(adjectives)
                
                results.append({
                    'model': model_name,
                    'precision': precision,
                    'avg_inference_ms': avg_time,
                    'avg_adjectives': avg_adjectives,
                    'speed_rank': len([r for r in results if r['avg_inference_ms'] < avg_time]) + 1,
                    'quality_rank': len([r for r in results if r['avg_adjectives'] > avg_adjectives]) + 1
                })
                
                print(f"   ✅ Avg inference: {avg_time:.1f}ms")
                print(f"   ✅ Avg adjectives: {avg_adjectives:.2f}")
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            continue
    
    # Analysis
    print("\n" + "="*60)
    print("📊 SPEED-QUALITY TRADE-OFF ANALYSIS")
    print("="*60)
    
    if results:
        df = pd.DataFrame(results)
        
        print("\n🏆 PERFORMANCE MATRIX:")
        for result in results:
            speed_stars = "⭐" * (4 - result['speed_rank'] + 1)
            quality_stars = "⭐" * (4 - result['quality_rank'] + 1)
            print(f"{result['model']:20} | Speed: {result['avg_inference_ms']:5.1f}ms {speed_stars:4} | "
                  f"Quality: {result['avg_adjectives']:4.2f} adj {quality_stars:4}")
        
        # Find best balanced model
        balanced_scores = []
        for result in results:
            # Normalize scores (lower time and higher adjectives are better)
            norm_time = 1 - (result['avg_inference_ms'] / max(r['avg_inference_ms'] for r in results))
            norm_quality = result['avg_adjectives'] / max(r['avg_adjectives'] for r in results)
            balanced_score = (norm_time + norm_quality) / 2
            balanced_scores.append((result['model'], balanced_score))
        
        best_balanced = max(balanced_scores, key=lambda x: x[1])
        print(f"\n🎯 BEST BALANCED MODEL: {best_balanced[0]} (score: {best_balanced[1]:.3f})")
        
        # Save results
        with open("benchmark_speed_quality.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: benchmark_speed_quality.json")

if __name__ == "__main__":
    benchmark_speed_quality()
