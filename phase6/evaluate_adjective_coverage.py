import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_adjective_usage():
    """Evaluate the model's adjective usage"""
    
    # Load your trained model
    model_path = "./adjective_complete_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Test prompts
    test_prompts = [
        "Describe this image:",
        "This picture shows",
        "Generate a caption for this scene:",
        "What do you see in this image:"
    ]
    
    # Common adjectives to check for
    target_adjectives = [
        "beautiful", "vibrant", "massive", "towering", "peaceful", "serene",
        "colorful", "energetic", "historic", "stunning", "lush", "chaotic"
    ]
    
    device = model.device
    adjective_counts = {adj: 0 for adj in target_adjectives}
    total_generations = 0
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=80,
                num_return_sequences=3,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        for output in outputs:
            generated = tokenizer.decode(output, skip_special_tokens=True)
            total_generations += 1
            
            # Count adjective usage
            for adj in target_adjectives:
                if adj in generated.lower():
                    adjective_counts[adj] += 1
            
            logger.info(f"📝 {prompt} → {generated}")
    
    # Calculate coverage
    logger.info("\n🎯 ADJECTIVE COVERAGE RESULTS:")
    total_adjective_uses = sum(adjective_counts.values())
    coverage_percentage = (total_adjective_uses / (total_generations * len(target_adjectives))) * 100
    
    for adj, count in adjective_counts.items():
        usage = (count / total_generations) * 100
        logger.info(f"   {adj}: {count} uses ({usage:.1f}%)")
    
    logger.info(f"📊 TOTAL ADJECTIVE COVERAGE: {coverage_percentage:.1f}%")
    logger.info(f"📊 AVERAGE ADJECTIVES PER DESCRIPTION: {total_adjective_uses/total_generations:.1f}")

if __name__ == "__main__":
    evaluate_adjective_usage()
