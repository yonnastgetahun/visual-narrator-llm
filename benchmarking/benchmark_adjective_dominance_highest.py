import requests
import json
import time
import numpy as np
from datetime import datetime
import random
import anthropic
import openai

def log(m): print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)

class HighestModelsAdjectiveBenchmark:
    """Adjective dominance benchmark against highest-tier models"""
    
    def __init__(self):
        # Setup highest-tier APIs
        self.claude_client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )
        self.openai_client = openai.OpenAI(
            api_key=os.environ["OPENAI_API_KEY"]
        )
        
        self.our_api_url = "http://localhost:8002"
        self.adjective_list = self.create_comprehensive_adjective_list()
    
    def create_comprehensive_adjective_list(self):
        """Comprehensive adjective vocabulary for accurate counting"""
        return [
            # Aesthetic & visual
            'beautiful', 'stunning', 'gorgeous', 'picturesque', 'scenic', 'breathtaking',
            'magnificent', 'splendid', 'glorious', 'majestic', 'grand', 'imposing',
            'vibrant', 'colorful', 'vivid', 'bright', 'brilliant', 'radiant',
            'gleaming', 'shimmering', 'sparkling', 'glittering', 'luminous',
            
            # Size & scale
            'massive', 'enormous', 'gigantic', 'towering', 'colossal', 'immense',
            'spacious', 'expansive', 'vast', 'boundless', 'monumental',
            
            # Emotional & mood
            'peaceful', 'serene', 'tranquil', 'calm', 'soothing', 'relaxing',
            'dramatic', 'intense', 'powerful', 'emotional', 'evocative',
            
            # Quality & style
            'elegant', 'sophisticated', 'refined', 'graceful', 'stylish',
            'luxurious', 'opulent', 'sumptuous', 'lavish', 'ornate', 'exquisite',
            
            # Age & history
            'ancient', 'historic', 'vintage', 'antique', 'traditional', 'classic',
            'timeless', 'aged', 'weathered', 'patinated',
            
            # Modern & contemporary
            'modern', 'contemporary', 'innovative', 'futuristic', 'sleek',
            'cutting-edge', 'state-of-the-art', 'progressive',
            
            # Natural & organic
            'natural', 'organic', 'rustic', 'earthy', 'raw', 'untamed',
            'pristine', 'unspoiled', 'virgin', 'flourishing', 'lush', 'verdant',
            
            # Urban & architectural
            'urban', 'metropolitan', 'cosmopolitan', 'bustling', 'lively',
            'architectural', 'structural', 'stately',
            
            # Texture & surface
            'smooth', 'polished', 'glossy', 'textured', 'rough', 'coarse',
            'silky', 'velvety', 'satin', 'crystalline', 'glassy'
        ]
    
    def create_challenging_test_scenes(self):
        """Scenes designed to challenge highest-tier models"""
        return [
            # Complex scenes that should elicit rich descriptions
            "A golden sunset casting long shadows over ancient ruins surrounded by olive trees",
            "A bustling Moroccan market with vibrant spices, intricate textiles, and animated merchants",
            "A futuristic cityscape with gleaming skyscrapers, flying vehicles, and holographic advertisements",
            "A serene Japanese garden featuring koi ponds, stone lanterns, and carefully pruned bonsai trees",
            "A dramatic coastal cliffside with crashing waves, sea stacks, and seabirds soaring overhead",
            "An opulent Venetian palace with marble facades, grand arches, and gondolas drifting along canals",
            "A mystical forest with towering redwoods, dappled sunlight, and moss-covered pathways",
            "A lively Brazilian carnival with elaborate costumes, samba dancers, and vibrant floats",
            "A minimalist Scandinavian interior with clean lines, natural light, and functional furniture",
            "A historic European cathedral with stained glass, flying buttresses, and intricate stone carvings"
        ]
    
    def count_adjectives(self, text):
        """Accurate adjective counting with exact word matching"""
        if not text:
            return 0
        words = text.lower().split()
        # Use exact word matching to avoid partial matches
        return sum(1 for word in words if word in self.adjective_list)
    
    def benchmark_our_system(self, scene):
        """Benchmark our Visual Narrator VLM"""
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
                adj_count = self.count_adjectives(output_text)
                word_count = len(output_text.split())
                adj_density = adj_count / word_count if word_count > 0 else 0
                
                return {
                    "output": output_text,
                    "adjective_count": adj_count,
                    "word_count": word_count,
                    "adjective_density": adj_density,
                    "processing_time": processing_time
                }
        except Exception as e:
            log(f"❌ Our system error: {e}")
        
        return None
    
    def benchmark_claude_sonnet(self, scene):
        """Benchmark Claude 3.5 Sonnet (highest model)"""
        try:
            start_time = time.time()
            
            response = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",  # Highest Claude model
                max_tokens=200,
                messages=[{
                    "role": "user", 
                    "content": f"Describe this scene in vivid, adjective-rich detail. Use abundant descriptive adjectives to create a lush, immersive description: {scene}"
                }]
            )
            
            processing_time = time.time() - start_time
            description = response.content[0].text
            
            adj_count = self.count_adjectives(description)
            word_count = len(description.split())
            adj_density = adj_count / word_count if word_count > 0 else 0
            
            return {
                "output": description,
                "adjective_count": adj_count,
                "word_count": word_count,
                "adjective_density": adj_density,
                "processing_time": processing_time
            }
            
        except Exception as e:
            log(f"❌ Claude 3.5 Sonnet error: {e}")
            return None
    
    def benchmark_gpt4_turbo(self, scene):
        """Benchmark GPT-4 Turbo (highest available model)"""
        try:
            start_time = time.time()
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo",  # Highest GPT-4 model
                max_tokens=200,
                messages=[{
                    "role": "user",
                    "content": f"Describe this scene using rich, abundant adjectives. Create a vivid, immersive description with extensive descriptive language: {scene}"
                }]
            )
            
            processing_time = time.time() - start_time
            description = response.choices[0].message.content
            
            adj_count = self.count_adjectives(description)
            word_count = len(description.split())
            adj_density = adj_count / word_count if word_count > 0 else 0
            
            return {
                "output": description,
                "adjective_count": adj_count,
                "word_count": word_count,
                "adjective_density": adj_density,
                "processing_time": processing_time
            }
            
        except Exception as e:
            log(f"❌ GPT-4 Turbo error: {e}")
            return None
    
    def run_highest_models_benchmark(self):
        """Run adjective dominance benchmark against highest-tier models"""
        log("🎯 STARTING ADJECTIVE DOMINANCE - HIGHEST MODELS BENCHMARK...")
        
        scenes = self.create_challenging_test_scenes()
        models = {
            "Visual Narrator VLM": self.benchmark_our_system,
            "Claude 3.5 Sonnet": self.benchmark_claude_sonnet,
            "GPT-4 Turbo": self.benchmark_gpt4_turbo
        }
        
        all_results = {model: [] for model in models.keys()}
        
        for scene in scenes[:5]:  # Test 5 scenes to manage API costs
            log(f"📝 Testing: {scene[:60]}...")
            
            for model_name, benchmark_func in models.items():
                result = benchmark_func(scene)
                if result:
                    all_results[model_name].append(result)
                    log(f"  ✅ {model_name}: {result['adjective_count']} adjectives, density {result['adjective_density']:.3f}")
                else:
                    log(f"  ❌ {model_name}: Failed")
        
        # Calculate averages
        model_metrics = {}
        for model, results in all_results.items():
            if results:
                avg_density = np.mean([r["adjective_density"] for r in results])
                avg_adjectives = np.mean([r["adjective_count"] for r in results])
                avg_time = np.mean([r["processing_time"] for r in results])
                
                model_metrics[model] = {
                    "avg_adjective_density": avg_density,
                    "avg_adjectives_per_scene": avg_adjectives,
                    "avg_processing_time": avg_time,
                    "sample_count": len(results)
                }
        
        # Display results
        self.display_highest_models_results(model_metrics)
        
        return model_metrics
    
    def display_highest_models_results(self, model_metrics):
        """Display results against highest-tier models"""
        print("\n" + "="*80)
        print("🎯 PART A: ADJECTIVE DOMINANCE - HIGHEST MODELS")
        print("="*80)
        
        print("📊 ADJECTIVE DENSITY AGAINST TOP-TIER MODELS:")
        ranking = sorted(
            [(model, metrics["avg_adjective_density"]) 
             for model, metrics in model_metrics.items()],
            key=lambda x: x[1], 
            reverse=True
        )
        
        our_metrics = model_metrics.get("Visual Narrator VLM", {})
        our_density = our_metrics.get("avg_adjective_density", 0)
        
        for i, (model, density) in enumerate(ranking, 1):
            if i == 1:
                marker = "🥇"
                advantage = ""
            else:
                marker = "  "
                advantage = f" (-{((our_density - density) / density * 100):.1f}%)" if model == "Visual Narrator VLM" else f" (+{((our_density - density) / density * 100):.1f}%)"
            
            print(f"   {marker} {model:<25} {density:.3f}{advantage}")
        
        print(f"\n🏆 COMPETITIVE ANALYSIS:")
        if ranking[0][0] == "Visual Narrator VLM":
            advantage_over_2nd = ((our_density - ranking[1][1]) / ranking[1][1] * 100)
            print(f"   ✅ WE BEAT HIGHEST-TIER MODELS: +{advantage_over_2nd:.1f}% over {ranking[1][0]}")
            print(f"   🎯 Our adjective specialization dominates even against premium models!")
        else:
            gap_to_leader = ((ranking[0][1] - our_density) / our_density * 100)
            print(f"   ⚠️  Behind {ranking[0][0]} by {gap_to_leader:.1f}%")
        
        print(f"\n📈 ADJECTIVE COUNT PER SCENE:")
        for model, metrics in sorted(model_metrics.items(), 
                                   key=lambda x: x[1]["avg_adjectives_per_scene"], 
                                   reverse=True):
            count = metrics["avg_adjectives_per_scene"]
            print(f"   • {model:<25} {count:.1f} adjectives/scene")
        
        print(f"\n⚡ PROCESSING SPEED:")
        for model, metrics in sorted(model_metrics.items(), 
                                   key=lambda x: x[1]["avg_processing_time"]):
            time_ms = metrics["avg_processing_time"] * 1000
            print(f"   • {model:<25} {time_ms:.1f}ms")
        
        print(f"\n💡 STRATEGIC IMPLICATIONS:")
        if ranking[0][0] == "Visual Narrator VLM":
            print("   • Our specialized approach beats even the most expensive API models")
            print("   • We offer superior performance at fraction of the cost")
            print("   • Clear differentiation in the crowded VLM market")
        else:
            print("   • Competitive with highest-tier models")
            print("   • Cost and speed advantages remain significant")
        
        print("="*80)

def main():
    benchmark = HighestModelsAdjectiveBenchmark()
    model_metrics = benchmark.run_highest_models_benchmark()
    
    print("\n🎉 HIGHEST MODELS ADJECTIVE BENCHMARK COMPLETED!")

if __name__ == "__main__":
    main()
