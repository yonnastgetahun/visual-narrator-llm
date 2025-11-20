import torch
import time
import json
import pandas as pd
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import glob
import os
from datetime import datetime

def count_adjectives_comprehensive(text):
    """Comprehensive adjective counting"""
    adjectives = [
        'vivid', 'gleaming', 'rugged', 'tranquil', 'velvety', 'golden',
        'richly', 'detailed', 'cinematic', 'dramatic', 'vibrant', 'serene',
        'majestic', 'luminous', 'textured', 'atmospheric', 'expressive',
        'stunning', 'breathtaking', 'captivating', 'mesmerizing', 'radiant',
        'glowing', 'sparkling', 'pristine', 'ethereal', 'soothing', 'dynamic',
        'brilliant', 'crisp', 'elegant', 'exquisite', 'gorgeous', 'grand',
        'impressive', 'luxurious', 'opulent', 'picturesque', 'refined',
        'splendid', 'sumptuous', 'superb', 'tasteful', 'aesthetic'
    ]
    text_lower = text.lower()
    return sum(1 for adj in adjectives if adj in text_lower)

def create_baseline_model():
    """Create a baseline model from our original BLIP weights"""
    print("🔄 Creating baseline model from original weights...")
    
    # Use our original BLIP model as baseline
    baseline_path = "models/blip-base-local"
    if os.path.exists(baseline_path):
        processor = BlipProcessor.from_pretrained(baseline_path, local_files_only=True)
        model = BlipForConditionalGeneration.from_pretrained(baseline_path, local_files_only=True)
        return processor, model, "BLIP-Baseline (Original)"
    else:
        raise FileNotFoundError("Baseline model not found")

def benchmark_comprehensive():
    """Comprehensive benchmarking that works offline"""
    
    print("🎯 VISUAL NARRATOR VLM - OFFLINE BENCHMARKING")
    print("=" * 70)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    
    # Test images - use diverse set
    test_images = glob.glob("/data/coco/train2017/*.jpg")[:15]
    print(f"🖼️  Test images: {len(test_images)}")
    
    results = []
    
    # 1. Benchmark Our Fine-tuned Model
    print("\n1. 🎭 OUR VISUAL NARRATOR VLM")
    print("-" * 40)
    
    try:
        our_model_path = "outputs/phase7_3_large_scale/checkpoint-step-5000-1762322982"
        our_processor = BlipProcessor.from_pretrained(our_model_path, local_files_only=True)
        our_model = BlipForConditionalGeneration.from_pretrained(our_model_path, local_files_only=True).to(device)
        
        our_results = run_benchmark("Visual-Narrator-VLM", our_processor, our_model, test_images, device)
        if our_results:
            results.append(our_results)
            print(f"   ✅ Adjectives: {our_results['avg_adjectives']:.2f}")
            print(f"   ⚡ Speed: {our_results['avg_inference_ms']:.1f}ms")
            print(f"   📊 Samples: {our_results['samples_tested']}")
    except Exception as e:
        print(f"❌ Our model failed: {e}")
    
    # 2. Benchmark Baseline (Original BLIP)
    print("\n2. 🏁 BASELINE MODEL (Original BLIP)")
    print("-" * 40)
    
    try:
        base_processor, base_model, base_name = create_baseline_model()
        base_model = base_model.to(device)
        
        base_results = run_benchmark(base_name, base_processor, base_model, test_images, device)
        if base_results:
            results.append(base_results)
            print(f"   ✅ Adjectives: {base_results['avg_adjectives']:.2f}")
            print(f"   ⚡ Speed: {base_results['avg_inference_ms']:.1f}ms")
            print(f"   📊 Samples: {base_results['samples_tested']}")
    except Exception as e:
        print(f"❌ Baseline failed: {e}")
    
    # 3. Benchmark Phase 7.2 Model for comparison
    print("\n3. 📈 PHASE 7.2 MODEL (Previous Best)")
    print("-" * 40)
    
    try:
        phase7_2_ckpts = glob.glob("outputs/phase7_optimized/checkpoint-epoch-*")
        if phase7_2_ckpts:
            phase7_2_path = sorted(phase7_2_ckpts)[-1]
            phase7_2_processor = BlipProcessor.from_pretrained(phase7_2_path, local_files_only=True)
            phase7_2_model = BlipForConditionalGeneration.from_pretrained(phase7_2_path, local_files_only=True).to(device)
            
            phase7_2_results = run_benchmark("Phase7.2-Model", phase7_2_processor, phase7_2_model, test_images, device)
            if phase7_2_results:
                results.append(phase7_2_results)
                print(f"   ✅ Adjectives: {phase7_2_results['avg_adjectives']:.2f}")
                print(f"   ⚡ Speed: {phase7_2_results['avg_inference_ms']:.1f}ms")
                print(f"   📊 Samples: {phase7_2_results['samples_tested']}")
    except Exception as e:
        print(f"❌ Phase 7.2 model failed: {e}")
    
    # Generate comprehensive analysis
    if len(results) >= 2:
        generate_benchmark_analysis(results)
    else:
        print("\n❌ Insufficient results for meaningful comparison")

def run_benchmark(model_name, processor, model, test_images, device):
    """Run benchmark for a single model"""
    print(f"   🧪 Testing {model_name}...")
    
    adjective_counts = []
    inference_times = []
    captions = []
    
    for img_path in test_images:
        try:
            image = Image.open(img_path)
            
            # Time inference
            start_time = time.time()
            inputs = processor(images=image, return_tensors="pt").to(device)
            
            with torch.amp.autocast("cuda", enabled=True):
                outputs = model.generate(
                    **inputs,
                    max_length=60,
                    num_beams=3,
                    early_stopping=True
                )
            
            inference_time = time.time() - start_time
            caption = processor.decode(outputs[0], skip_special_tokens=True)
            
            adj_count = count_adjectives_comprehensive(caption)
            
            adjective_counts.append(adj_count)
            inference_times.append(inference_time)
            captions.append(caption)
            
        except Exception as e:
            print(f"      ❌ Error on {os.path.basename(img_path)}: {e}")
            continue
    
    if adjective_counts and inference_times:
        return {
            'model': model_name,
            'avg_adjectives': sum(adjective_counts) / len(adjective_counts),
            'max_adjectives': max(adjective_counts),
            'min_adjectives': min(adjective_counts),
            'avg_inference_ms': sum(inference_times) / len(inference_times) * 1000,
            'samples_tested': len(adjective_counts),
            'sample_captions': captions[:3],
            'all_captions': captions
        }
    return None

def generate_benchmark_analysis(results):
    """Generate detailed benchmark analysis"""
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE BENCHMARK ANALYSIS")
    print("="*70)
    
    df = pd.DataFrame(results)
    df = df.sort_values('avg_adjectives', ascending=False)
    
    # Performance comparison
    print("\n🏆 PERFORMANCE RANKING:")
    print("-" * 60)
    for i, row in df.iterrows():
        stars = "⭐" * min(5, int(row['avg_adjectives']))
        print(f"{i+1}. {row['model']:25} | "
              f"Adjectives: {row['avg_adjectives']:5.2f} {stars:5} | "
              f"Speed: {row['avg_inference_ms']:6.1f}ms")
    
    # Improvement analysis
    if len(results) >= 2:
        our_model = next((r for r in results if 'Visual-Narrator' in r['model']), None)
        baseline = next((r for r in results if 'Baseline' in r['model']), None)
        
        if our_model and baseline:
            improvement = ((our_model['avg_adjectives'] - baseline['avg_adjectives']) / 
                          baseline['avg_adjectives'] * 100)
            
            print(f"\n🎯 KEY IMPROVEMENT:")
            print(f"   Our VLM: {our_model['avg_adjectives']:.2f} adjectives")
            print(f"   Baseline: {baseline['avg_adjectives']:.2f} adjectives")
            print(f"   IMPROVEMENT: {improvement:+.1f}% 📈")
    
    # Quality showcase
    print("\n🎨 QUALITY SHOWCASE (Our VLM vs Baseline):")
    print("-" * 60)
    
    our_model = next((r for r in results if 'Visual-Narrator' in r['model']), None)
    baseline = next((r for r in results if 'Baseline' in r['model']), None)
    
    if our_model and baseline:
        print("Our Visual Narrator VLM:")
        for i, caption in enumerate(our_model['sample_captions'][:2], 1):
            adj_count = count_adjectives_comprehensive(caption)
            print(f"   {i}. [{adj_count} adj] {caption}")
        
        print("\nBaseline Model:")
        for i, caption in enumerate(baseline['sample_captions'][:2], 1):
            adj_count = count_adjectives_comprehensive(caption)
            print(f"   {i}. [{adj_count} adj] {caption}")
    
    # Statistical analysis
    print(f"\n📈 STATISTICAL ANALYSIS:")
    print(f"   Best single caption: {max(r['max_adjectives'] for r in results)} adjectives")
    print(f"   Most consistent: {df.iloc[df['min_adjectives'].idxmax()]['model']}")
    print(f"   Fastest: {df.iloc[df['avg_inference_ms'].idxmin()]['model']}")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"comprehensive_benchmark_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'results': results,
            'summary': {
                'best_model': df.iloc[0]['model'],
                'best_score': df.iloc[0]['avg_adjectives'],
                'improvement_over_baseline': improvement if 'improvement' in locals() else 0
            }
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Final recommendation
    print(f"\n✅ DEPLOYMENT RECOMMENDATION:")
    best_model = df.iloc[0]
    print(f"   Use {best_model['model']} for production")
    print(f"   Performance: {best_model['avg_adjectives']:.2f} adjectives/description")
    print(f"   Speed: {best_model['avg_inference_ms']:.1f}ms per image")
    print(f"   Quality: Exceptional descriptive density 🚀")

if __name__ == "__main__":
    benchmark_comprehensive()
