import requests
import json
import time
import numpy as np
from datetime import datetime
import anthropic
import openai

def log(m): print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)

class RealClaudeBenchmark:
    """Real benchmark using actual Claude API calls"""
    
    def __init__(self):
        # Use your actual API key for real testing
        self.claude_client = anthropic.Anthropic(
            api_key="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"
        )
        self.openai_client = openai.OpenAI(
            api_key="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"
        )
        
        self.our_api_url = "http://localhost:8002"
        
        # Test which Claude models actually work
        self.available_models = self.test_claude_models()
    
    def test_claude_models(self):
        """Test which Claude models are actually available"""
        log("🔍 TESTING CLAUDE MODEL AVAILABILITY...")
        
        test_models = [
            "claude-3-5-sonnet-20241022",  # Try the 4.x version
            "claude-3-opus-20240229",       # Fallback Opus
            "claude-3-haiku-20240307",      # Fallback Haiku
            "claude-3-5-sonnet-latest",     # Try latest tag
            "claude-3-sonnet-20240229"      # Legacy Sonnet
        ]
        
        available = []
        for model in test_models:
            try:
                response = self.claude_client.messages.create(
                    model=model,
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Say hello"}]
                )
                available.append(model)
                log(f"✅ {model}: AVAILABLE")
            except Exception as e:
                log(f"❌ {model}: {str(e)[:100]}...")
        
        return available
    
    def benchmark_real_claude(self, scene, model_name):
        """Real Claude API benchmark"""
        try:
            start_time = time.time()
            
            response = self.claude_client.messages.create(
                model=model_name,
                max_tokens=150,
                messages=[{
                    "role": "user", 
                    "content": f"Describe this scene vividly with rich adjectives: {scene}"
                }]
            )
            
            processing_time = time.time() - start_time
            description = response.content[0].text
            
            # Count adjectives
            adjectives = ['beautiful', 'vibrant', 'majestic', 'serene', 'elegant', 'dramatic', 
                         'stunning', 'gorgeous', 'picturesque', 'breathtaking']
            adj_count = sum(1 for adj in adjectives if adj in description.lower())
            word_count = len(description.split())
            adj_density = adj_count / word_count if word_count > 0 else 0
            
            return {
                "output": description,
                "adjective_count": adj_count,
                "word_count": word_count,
                "adjective_density": adj_density,
                "processing_time": processing_time,
                "model": f"Claude ({model_name})"
            }
            
        except Exception as e:
            log(f"❌ Claude {model_name} error: {e}")
            return None
    
    def benchmark_real_gpt4(self, scene):
        """Real GPT-4 API benchmark"""
        try:
            start_time = time.time()
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo",
                max_tokens=150,
                messages=[{
                    "role": "user",
                    "content": f"Describe this scene vividly with rich adjectives: {scene}"
                }]
            )
            
            processing_time = time.time() - start_time
            description = response.choices[0].message.content
            
            adjectives = ['beautiful', 'vibrant', 'majestic', 'serene', 'elegant', 'dramatic']
            adj_count = sum(1 for adj in adjectives if adj in description.lower())
            word_count = len(description.split())
            adj_density = adj_count / word_count if word_count > 0 else 0
            
            return {
                "output": description,
                "adjective_count": adj_count,
                "word_count": word_count,
                "adjective_density": adj_density,
                "processing_time": processing_time,
                "model": "GPT-4 Turbo"
            }
            
        except Exception as e:
            log(f"❌ GPT-4 error: {e}")
            return None
    
    def benchmark_our_system(self, scene):
        """Our system benchmark"""
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.our_api_url}/describe/scene",
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
                output_text = result["enhanced_description"]
                
                adjectives = ['beautiful', 'vibrant', 'majestic', 'serene', 'elegant', 'dramatic']
                adj_count = sum(1 for adj in adjectives if adj in output_text.lower())
                word_count = len(output_text.split())
                adj_density = adj_count / word_count if word_count > 0 else 0
                
                return {
                    "output": output_text,
                    "adjective_count": adj_count,
                    "word_count": word_count,
                    "adjective_density": adj_density,
                    "processing_time": processing_time,
                    "model": "Visual Narrator VLM"
                }
        except Exception as e:
            log(f"❌ Our system error: {e}")
        
        return None
    
    def run_real_comparison(self):
        """Run real API comparison with highest available models"""
        log("🎯 STARTING REAL API COMPARISON...")
        
        test_scenes = [
            "A golden sunset over ancient ruins with olive trees surrounding",
            "A bustling market with vibrant spices and colorful textiles",
            "A futuristic cityscape with gleaming skyscrapers and flying vehicles"
        ]
        
        all_results = []
        
        # Use highest available Claude model
        claude_model = self.available_models[0] if self.available_models else "claude-3-opus-20240229"
        log(f"🎯 USING HIGHEST CLAUDE MODEL: {claude_model}")
        
        for scene in test_scenes[:2]:  # Limit to 2 scenes to manage costs
            log(f"📝 Testing: {scene[:60]}...")
            
            # Our system
            our_result = self.benchmark_our_system(scene)
            if our_result:
                all_results.append(our_result)
                log(f"  ✅ Our System: {our_result['adjective_count']} adjectives, {our_result['processing_time']:.3f}s")
            
            # Real Claude
            claude_result = self.benchmark_real_claude(scene, claude_model)
            if claude_result:
                all_results.append(claude_result)
                log(f"  ✅ Claude: {claude_result['adjective_count']} adjectives, {claude_result['processing_time']:.3f}s")
            
            # Real GPT-4
            gpt_result = self.benchmark_real_gpt4(scene)
            if gpt_result:
                all_results.append(gpt_result)
                log(f"  ✅ GPT-4: {gpt_result['adjective_count']} adjectives, {gpt_result['processing_time']:.3f}s")
        
        # Generate comparison report
        self.generate_real_comparison_report(all_results, claude_model)
        
        return all_results
    
    def generate_real_comparison_report(self, results, claude_model):
        """Generate report with real API results"""
        print("\n" + "="*80)
        print("🎯 REAL API COMPARISON - HIGHEST AVAILABLE MODELS")
        print("="*80)
        
        model_results = {}
        for result in results:
            model = result["model"]
            if model not in model_results:
                model_results[model] = []
            model_results[model].append(result)
        
        print("📊 REAL PERFORMANCE COMPARISON:")
        print("-" * 80)
        
        for model, data in model_results.items():
            avg_density = np.mean([r["adjective_density"] for r in data])
            avg_time = np.mean([r["processing_time"] for r in data])
            avg_adjectives = np.mean([r["adjective_count"] for r in data])
            
            print(f"\n🔍 {model}:")
            print(f"   • Adjective Density: {avg_density:.3f}")
            print(f"   • Avg Adjectives/Scene: {avg_adjectives:.1f}")
            print(f"   • Processing Time: {avg_time*1000:.1f}ms")
        
        print(f"\n🏆 REAL-WORLD ADVANTAGES:")
        our_data = model_results.get("Visual Narrator VLM", [])
        claude_data = model_results.get(f"Claude ({claude_model})", [])
        
        if our_data and claude_data:
            our_density = np.mean([r["adjective_density"] for r in our_data])
            claude_density = np.mean([r["adjective_density"] for r in claude_data])
            our_time = np.mean([r["processing_time"] for r in our_data])
            claude_time = np.mean([r["processing_time"] for r in claude_data])
            
            density_advantage = ((our_density - claude_density) / claude_density * 100)
            speed_advantage = ((claude_time - our_time) / our_time)
            
            print(f"   • Adjective Advantage: +{density_advantage:.1f}% over Claude")
            print(f"   • Speed Advantage: {speed_advantage:.0f}x faster than Claude")
            print(f"   • Cost Advantage: Local vs. API pricing")
        
        print(f"\n💡 STRATEGIC POSITIONING:")
        print(f"   • Tested against real Claude: {claude_model}")
        print(f"   • Real API calls with actual performance data")
        print(f"   • Cost-effective comparison (limited test scenes)")
        print("="*80)

def main():
    benchmark = RealClaudeBenchmark()
    results = benchmark.run_real_comparison()
    
    print("\n🎉 REAL API BENCHMARK COMPLETED!")
    print("📈 Using actual Claude and GPT-4 API calls!")

if __name__ == "__main__":
    main()
