import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelComparisonBenchmark:
    """Compare all our trained models to find the best performer"""
    
    def __init__(self):
        self.test_prompts = [
            "Describe this urban scene:",
            "This beautiful landscape shows",
            "Generate caption for historic building",
            "What do you see in this city",
            "A peaceful natural scene with",
            "Modern architecture featuring",
            "Ancient ruins showing",
            "Colorful street art depicting"
        ]
        
        self.adjective_list = [
            'beautiful', 'vibrant', 'colorful', 'massive', 'enormous', 'gigantic',
            'peaceful', 'serene', 'tranquil', 'calm', 'chaotic', 'busy', 'bustling',
            'lively', 'dramatic', 'intense', 'powerful', 'emotional', 'elegant',
            'sophisticated', 'refined', 'ancient', 'historic', 'traditional', 'classic',
            'modern', 'contemporary', 'innovative', 'natural', 'organic', 'rustic',
            'urban', 'metropolitan', 'cosmopolitan', 'stunning', 'magnificent',
            'spectacular', 'quiet', 'bright', 'vivid', 'brilliant', 'dark', 'shadowy',
            'mysterious', 'warm', 'inviting', 'cozy', 'comfortable', 'cold', 'stark'
        ]
        
        self.models_to_test = [
            ("phase6/adjective_complete_model", "Phase 6.1 Complete"),
            ("phase6/phase6_2_focused_adjectives", "Phase 6.2 Focused"),
            ("phase6/adjective_dominance_full", "Adjective Dominance Full"),
            ("phase6/adjective_model_proper", "Adjective Model Proper"),
            ("base", "OPT-2.7B Base")  # Base model for comparison
        ]
    
    def load_model(self, model_path, model_name):
        """Load a specific model"""
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if model_path == "base":
            model = AutoModelForCausalLM.from_pretrained(
                "facebook/opt-2.7b",
                torch_dtype=torch.float16,
                device_map="auto"
            )
            return model, tokenizer, "base"
        
        if os.path.exists(model_path):
            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    "facebook/opt-2.7b",
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                model = PeftModel.from_pretrained(base_model, model_path)
                logger.info(f"✅ Loaded {model_name} successfully")
                return model, tokenizer, "fine-tuned"
            except Exception as e:
                logger.warning(f"❌ Failed to load {model_name}: {e}")
                return None, None, "error"
        else:
            logger.warning(f"❌ Model path not found: {model_path}")
            return None, None, "not_found"
    
    def count_adjectives(self, text):
        """Count adjectives in generated text"""
        text_lower = text.lower()
        return sum(1 for adj in self.adjective_list if adj in text_lower)
    
    def benchmark_model(self, model_path, model_name):
        """Benchmark a single model"""
        logger.info(f"🧪 BENCHMARKING: {model_name}")
        
        model, tokenizer, model_type = self.load_model(model_path, model_name)
        if model is None:
            return None
        
        model.eval()
        results = []
        total_adjectives = 0
        
        for prompt in self.test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=80,
                    do_sample=True,
                    temperature=0.8,
                    repetition_penalty=1.2,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            adj_count = self.count_adjectives(generated)
            total_adjectives += adj_count
            
            results.append({
                "prompt": prompt,
                "generated": generated,
                "adjectives_count": adj_count
            })
        
        avg_adjectives = total_adjectives / len(self.test_prompts)
        
        return {
            "model_name": model_name,
            "model_path": model_path,
            "model_type": model_type,
            "average_adjectives": avg_adjectives,
            "total_adjectives": total_adjectives,
            "results": results
        }
    
    def run_comparison(self):
        """Run comparison across all models"""
        logger.info("🎯 COMPREHENSIVE MODEL COMPARISON BENCHMARK")
        logger.info("📊 Testing all available models to find best performer")
        
        comparison_results = {}
        
        for model_path, model_name in self.models_to_test:
            result = self.benchmark_model(model_path, model_name)
            if result:
                comparison_results[model_name] = result
                logger.info(f"   📈 {model_name}: {result['average_adjectives']:.2f} adjectives/description")
        
        # Sort by performance
        sorted_results = sorted(
            comparison_results.items(),
            key=lambda x: x[1]['average_adjectives'],
            reverse=True
        )
        
        logger.info("🎉 COMPARISON COMPLETE!")
        logger.info("🏆 PERFORMANCE RANKING:")
        for i, (name, result) in enumerate(sorted_results, 1):
            logger.info(f"   {i}. {name}: {result['average_adjectives']:.2f} adjectives/description")
        
        # Save detailed results
        with open("phase6.3/benchmarks/model_comparison_results.json", "w") as f:
            json.dump(comparison_results, f, indent=2)
        
        return sorted_results

if __name__ == "__main__":
    benchmark = ModelComparisonBenchmark()
    results = benchmark.run_comparison()
    
    print("\n" + "="*60)
    if results:
        best_model = results[0]
        best_score = best_model[1]['average_adjectives']
        print(f"🏆 BEST MODEL: {best_model[0]}")
        print(f"📈 SCORE: {best_score:.2f} adjectives/description")
        
        if best_score >= 5.0:
            print("🎉 PHASE 6.2 TARGET ACHIEVED!")
        else:
            print("🔄 NEED TO RETRAIN - Best model below 5.0 target")
    print("="*60)
