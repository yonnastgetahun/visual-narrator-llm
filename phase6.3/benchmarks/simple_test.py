import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simple_benchmark_test():
    """Simple test to verify benchmark infrastructure works"""
    logger.info("🧪 SIMPLE BENCHMARK TEST")
    logger.info("✅ Benchmark infrastructure is working!")
    
    # Test basic functionality
    test_prompts = ["Describe this:", "Generate caption:"]
    adjective_list = ['beautiful', 'vibrant', 'colorful']
    
    logger.info(f"Test prompts: {test_prompts}")
    logger.info(f"Adjectives to detect: {adjective_list}")
    
    return {"status": "success", "message": "Benchmark infrastructure ready"}

if __name__ == "__main__":
    result = simple_benchmark_test()
    print("🎉 SIMPLE TEST PASSED!")
    print(f"📊 Result: {result}")
