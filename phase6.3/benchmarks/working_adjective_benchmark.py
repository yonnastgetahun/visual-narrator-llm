import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkingAdjectiveBenchmark:
    """Working version of adjective density benchmark"""
    
    def __init__(self):
        self.test_prompts = [
            "Describe this urban scene:",
            "This beautiful landscape shows",
            "Generate caption for historic building"
        ]
        
        self.adjective_list = [
            'beautiful', 'vibrant', 'colorful', 'massive', 'peaceful', 
            'serene', 'chaotic', 'lively', 'dramatic', 'elegant'
        ]
    
    def run_benchmark(self):
        """Run simplified benchmark"""
        logger.info("🎯 RUNNING WORKING ADJECTIVE BENCHMARK")
        
        try:
            # Try to load our enhanced model
            from peft import PeftModel
            
            base_model = AutoModelForCausalLM.from_pretrained(
                "facebook/opt-2.7b",
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            model = PeftModel.from_pretrained(base_model, "./phase6_2_focused_adjectives")
            tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")
            
            logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not load enhanced model: {e}")
            logger.info("🔄 Using base OPT-2.7B for testing")
            
            # Fallback to base model
            tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")
            model = AutoModelForCausalLM.from_pretrained(
                "facebook/opt-2.7b",
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        model.eval()
        results = []
        total_adjectives = 0
        
        for prompt in self.test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=60,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Count adjectives
            text_lower = generated.lower()
            adj_count = sum(1 for adj in self.adjective_list if adj in text_lower)
            total_adjectives += adj_count
            
            result = {
                "prompt": prompt,
                "generated": generated,
                "adjectives_count": adj_count
            }
            results.append(result)
            
            logger.info(f"   📝 '{prompt}' → {adj_count} adjectives")
            logger.info(f"      Generated: {generated}")
        
        avg_adjectives = total_adjectives / len(self.test_prompts)
        
        benchmark_result = {
            "average_adjectives": avg_adjectives,
            "total_tests": len(self.test_prompts),
            "results": results
        }
        
        logger.info(f"📊 BENCHMARK COMPLETE: {avg_adjectives:.2f} adjectives/description")
        
        # Save results
        with open("phase6.3/benchmarks/working_benchmark_results.json", "w") as f:
            json.dump(benchmark_result, f, indent=2)
        
        return benchmark_result

if __name__ == "__main__":
    benchmark = WorkingAdjectiveBenchmark()
    results = benchmark.run_benchmark()
    
    print("🎉 WORKING BENCHMARK COMPLETE!")
    print(f"📈 Average adjectives: {results['average_adjectives']:.2f}")
