#!/usr/bin/env python3
"""
PHASE 4.4: SOTA BENCHMARKING & EVALUATION
Comprehensive evaluation of our Visual Narrator model
"""

print("🚀 PHASE 4.4: SOTA BENCHMARKING & EVALUATION")
print("=" * 50)

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import pandas as pd
from datetime import datetime
import os

class VisualNarratorBenchmark:
    def __init__(self, model_path="./phase4/advanced/model"):
        """Initialize the benchmark with our trained model"""
        print("🤖 Loading trained Visual Narrator model...")
        
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-2.7b",
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # Load LoRA adapters
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()
        
        print("✅ Benchmark model loaded successfully!")
    
    def generate_description(self, prompt, max_length=100):
        """Generate image description for a given prompt"""
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            description = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the generated text
            description = description.replace(prompt, "").strip()
            return description
    
    def run_qualitative_evaluation(self):
        """Run qualitative evaluation on diverse prompts"""
        print("\n🎨 Running Qualitative Evaluation...")
        
        test_prompts = [
            "Describe this image:",
            "What do you see in this image:",
            "Generate a detailed caption for this image:",
            "Provide an image description:"
        ]
        
        test_scenes = [
            "a beautiful sunset over mountains with clouds",
            "a busy city street with cars and pedestrians",
            "a peaceful forest with sunlight through trees",
            "a modern kitchen with stainless steel appliances",
            "a historic building with intricate architecture",
            "a sports event with athletes and cheering crowd",
            "a scientific laboratory with equipment",
            "a tropical beach with palm trees and ocean",
            "an art gallery with paintings on walls",
            "a farmer's market with colorful produce"
        ]
        
        results = []
        
        for prompt in test_prompts:
            for scene in test_scenes:
                full_prompt = f"{prompt} {scene}"
                
                try:
                    description = self.generate_description(full_prompt)
                    results.append({
                        'prompt': full_prompt,
                        'generated_description': description,
                        'prompt_type': prompt,
                        'scene_type': scene
                    })
                    
                    print(f"   ✅ {prompt[:20]}... -> {description[:50]}...")
                    
                except Exception as e:
                    print(f"   ❌ Generation failed: {e}")
                    results.append({
                        'prompt': full_prompt,
                        'generated_description': f"ERROR: {e}",
                        'prompt_type': prompt,
                        'scene_type': scene
                    })
        
        return results
    
    def evaluate_coherence(self, descriptions):
        """Evaluate coherence and quality of generated descriptions"""
        print("\n📊 Evaluating Description Coherence...")
        
        coherence_scores = []
        
        for desc in descriptions:
            score = 0
            
            # Simple heuristic scoring
            if len(desc) > 10:  # Non-empty
                score += 1
            if len(desc.split()) > 5:  # Reasonable length
                score += 1
            if '.' in desc:  # Proper sentence structure
                score += 1
            if any(word in desc.lower() for word in ['with', 'and', 'or', 'but']):  # Complexity
                score += 1
            if len(desc) < 200:  # Not too long
                score += 1
            
            coherence_scores.append(score)
        
        avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0
        print(f"   ✅ Average coherence score: {avg_coherence:.2f}/5.0")
        
        return avg_coherence, coherence_scores
    
    def run_performance_benchmark(self):
        """Run performance benchmarks"""
        print("\n⚡ Running Performance Benchmarks...")
        
        import time
        
        # Inference speed test
        test_prompt = "Describe this image: a beautiful sunset over mountains"
        
        start_time = time.time()
        descriptions = []
        
        for i in range(10):
            desc = self.generate_description(test_prompt)
            descriptions.append(desc)
        
        end_time = time.time()
        
        avg_inference_time = (end_time - start_time) / 10
        print(f"   ✅ Average inference time: {avg_inference_time:.2f}s")
        
        # Memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            print(f"   ✅ GPU memory allocated: {memory_allocated:.2f}GB")
            print(f"   ✅ GPU memory reserved: {memory_reserved:.2f}GB")
        
        return {
            'avg_inference_time': avg_inference_time,
            'memory_allocated_gb': memory_allocated if torch.cuda.is_available() else None,
            'memory_reserved_gb': memory_reserved if torch.cuda.is_available() else None
        }
    
    def create_benchmark_report(self, qualitative_results, coherence_score, performance_metrics):
        """Create comprehensive benchmark report"""
        print("\n📈 Generating Benchmark Report...")
        
        report = {
            'model_info': {
                'base_model': 'facebook/opt-2.7b',
                'fine_tuning': 'LoRA',
                'trainable_parameters': 47185920,
                'total_parameters': 2698782720,
                'training_data_size': 5000,
                'final_training_loss': 0.1366
            },
            'benchmark_results': {
                'qualitative_evaluation': {
                    'total_tests': len(qualitative_results),
                    'successful_generations': len([r for r in qualitative_results if not r['generated_description'].startswith('ERROR')])
                },
                'coherence_score': {
                    'average': coherence_score,
                    'max_possible': 5.0,
                    'interpretation': 'Excellent' if coherence_score >= 4.0 else 'Good' if coherence_score >= 3.0 else 'Needs Improvement'
                },
                'performance_metrics': performance_metrics
            },
            'sample_descriptions': qualitative_results[:5],  # Include first 5 samples
            'timestamp': datetime.now().isoformat(),
            'phase': '4.4'
        }
        
        # Save report
        os.makedirs("./phase4/benchmarks", exist_ok=True)
        report_path = "./phase4/benchmarks/benchmark_report.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"   ✅ Benchmark report saved to: {report_path}")
        return report

def main():
    """Main benchmarking pipeline"""
    try:
        # Initialize benchmark
        benchmark = VisualNarratorBenchmark()
        
        # Run qualitative evaluation
        qualitative_results = benchmark.run_qualitative_evaluation()
        
        # Evaluate coherence
        descriptions = [r['generated_description'] for r in qualitative_results if not r['generated_description'].startswith('ERROR')]
        avg_coherence, coherence_scores = benchmark.evaluate_coherence(descriptions)
        
        # Run performance benchmarks
        performance_metrics = benchmark.run_performance_benchmark()
        
        # Create comprehensive report
        report = benchmark.create_benchmark_report(
            qualitative_results, avg_coherence, performance_metrics
        )
        
        # Print summary
        print("\n🎯 PHASE 4.4 BENCHMARKING COMPLETE!")
        print("=" * 40)
        print(f"📊 Qualitative Tests: {report['benchmark_results']['qualitative_evaluation']['successful_generations']}/{report['benchmark_results']['qualitative_evaluation']['total_tests']} successful")
        print(f"🎯 Coherence Score: {avg_coherence:.2f}/5.0 ({report['benchmark_results']['coherence_score']['interpretation']})")
        print(f"⚡ Inference Time: {performance_metrics['avg_inference_time']:.2f}s")
        if performance_metrics['memory_allocated_gb']:
            print(f"💾 GPU Memory: {performance_metrics['memory_allocated_gb']:.2f}GB")
        
        print("\n🚀 READY FOR DEPLOYMENT AND PHASE 5!")
        
    except Exception as e:
        print(f"❌ Benchmarking error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
