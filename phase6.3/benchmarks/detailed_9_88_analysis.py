import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DetailedAnalysis:
    """Detailed analysis to understand the 9.88 vs 5.00 discrepancy"""
    
    def __init__(self):
        # Use the EXACT same test prompts from Phase 6.2 that gave 9.88
        self.phase6_2_prompts = [
            "Describe this urban scene:",
            "This beautiful landscape shows",
            "Generate caption for historic building", 
            "What do you see in this city",
            "A peaceful natural scene with",
            "Modern architecture featuring",
            "Ancient ruins showing",
            "Colorful street art depicting"
        ]
        
        # Expanded adjective list matching Phase 6.2
        self.adjective_list = [
            'beautiful', 'vibrant', 'colorful', 'massive', 'enormous', 'gigantic',
            'peaceful', 'serene', 'tranquil', 'calm', 'chaotic', 'busy', 'bustling',
            'lively', 'dramatic', 'intense', 'powerful', 'emotional', 'elegant',
            'sophisticated', 'refined', 'ancient', 'historic', 'traditional', 'classic',
            'modern', 'contemporary', 'innovative', 'natural', 'organic', 'rustic',
            'urban', 'metropolitan', 'cosmopolitan', 'stunning', 'magnificent',
            'spectacular', 'quiet', 'bright', 'vivid', 'brilliant', 'dark', 'shadowy',
            'mysterious', 'warm', 'inviting', 'cozy', 'comfortable', 'cold', 'stark',
            'luxurious', 'opulent', 'extravagant', 'simple', 'minimal', 'clean',
            'complex', 'intricate', 'expansive', 'vast', 'dynamic', 'energetic',
            'picturesque', 'idyllic', 'majestic', 'gorgeous', 'breathtaking', 'dazzling'
        ]
    
    def load_best_model(self):
        """Load our best performing model"""
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-2.7b",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        model = PeftModel.from_pretrained(base_model, "phase6/adjective_complete_model")
        logger.info("✅ Loaded Phase 6.1 Complete model")
        return model, tokenizer
    
    def count_adjectives_strict(self, text):
        """Count adjectives with strict matching"""
        text_lower = text.lower()
        count = 0
        found_adjectives = []
        
        for adj in self.adjective_list:
            if adj in text_lower:
                count += 1
                found_adjectives.append(adj)
        
        return count, found_adjectives
    
    def run_detailed_analysis(self):
        """Run detailed analysis with the same conditions as Phase 6.2"""
        logger.info("🔍 DETAILED 9.88 vs 5.00 ANALYSIS")
        logger.info("📊 Using Phase 6.1 Complete model with expanded adjective list")
        
        model, tokenizer = self.load_best_model()
        model.eval()
        
        detailed_results = []
        total_adjectives = 0
        
        for i, prompt in enumerate(self.phase6_2_prompts):
            # Use same generation parameters as Phase 6.2
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=80,
                    do_sample=True,
                    temperature=0.8,  # Same as Phase 6.2
                    repetition_penalty=1.2,  # Same as Phase 6.2
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            adj_count, found_adjs = self.count_adjectives_strict(generated)
            total_adjectives += adj_count
            
            result = {
                "prompt": prompt,
                "generated": generated,
                "adjectives_count": adj_count,
                "adjectives_found": found_adjs,
                "generation_length": len(generated)
            }
            detailed_results.append(result)
            
            logger.info(f"   {i+1}. '{prompt}'")
            logger.info(f"      → {adj_count} adjectives: {found_adjs}")
            logger.info(f"      Generated: {generated}")
            logger.info("")
        
        avg_adjectives = total_adjectives / len(self.phase6_2_prompts)
        
        analysis_result = {
            "model": "Phase 6.1 Complete",
            "average_adjectives": avg_adjectives,
            "total_adjectives": total_adjectives,
            "adjective_list_size": len(self.adjective_list),
            "detailed_results": detailed_results
        }
        
        logger.info(f"📊 FINAL ANALYSIS RESULTS:")
        logger.info(f"   Average adjectives: {avg_adjectives:.2f}")
        logger.info(f"   Total adjectives found: {total_adjectives}")
        logger.info(f"   Adjective list size: {len(self.adjective_list)}")
        logger.info(f"   Phase 6.2 Claim: 9.88")
        logger.info(f"   Current Measurement: {avg_adjectives:.2f}")
        
        # Save detailed analysis
        with open("phase6.3/benchmarks/detailed_analysis_results.json", "w") as f:
            json.dump(analysis_result, f, indent=2)
        
        return analysis_result

if __name__ == "__main__":
    analyzer = DetailedAnalysis()
    results = analyzer.run_detailed_analysis()
    
    print("\n" + "="*70)
    print("🔍 ROOT CAUSE ANALYSIS:")
    print(f"📈 Current measurement: {results['average_adjectives']:.2f} adjectives/description")
    print(f"🎯 Phase 6.2 claim: 9.88 adjectives/description")
    print(f"📊 Difference: {9.88 - results['average_adjectives']:.2f}")
    
    if results['average_adjectives'] >= 5.0:
        print("✅ CONFIRMED: Model meets Phase 6.2 target of 5.0+")
    else:
        print("❌ DISCREPANCY: Model below Phase 6.2 target")
    
    print("="*70)
