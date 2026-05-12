#!/usr/bin/env python3
"""
REAL SCORES BENCHMARK
- Uses actual Claude & OpenAI APIs
- Calculates real semantic accuracy
- No hard-coded scores
"""

import time
import requests
import anthropic
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
import statistics

class RealScoresBenchmark:
    def __init__(self):
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize with real API keys
        self.claude_client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )
        
        self.openai_client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"]
        )
        
        self.claude_opus_id = "claude-3-opus-20240229"
        self.gpt4_turbo_id = "gpt-4-turbo-preview"
        
        # Ground truth descriptions for semantic accuracy
        self.ground_truths = {
            "car_city": "A car is driving through a vibrant city at night with colorful neon lights reflecting on wet streets",
            "person_dancing": "A person is dancing energetically in a room with dynamic colorful lighting effects and moving shadows"
        }
    
    def calculate_semantic_accuracy(self, text1, text2):
        """Calculate real semantic similarity"""
        emb1 = self.semantic_model.encode([text1])[0]
        emb2 = self.semantic_model.encode([text2])[0]
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    
    def assess_narrative_quality(self, text):
        """Real narrative quality assessment"""
        if not text or len(text.strip()) < 8:
            return 0.0
        
        checks = [
            text[0].isupper(),
            text.endswith('.'),
            8 <= len(text.split()) <= 30,
            any(connector in text.lower() for connector in ['through', 'with', 'under', 'across', 'against', 'amidst']),
            any(cinematic in text.lower() for cinematic in ['sleek', 'vibrant', 'illuminated', 'graceful', 'dramatic', 'colorful']),
            '  ' not in text,
            text.count('.') <= 3,
            not any(broken in text.lower() for broken in [' a a ', ' the the ', 'neon neon'])
        ]
        
        return sum(checks) / len(checks)
    
    def test_our_system(self, scene_description, scene_key):
        """Test our polished system"""
        start_time = time.time()
        try:
            response = requests.post(
                "http://localhost:8008/describe/scene",
                json={
                    "scene_description": scene_description,
                    "enhance_adjectives": True
                },
                timeout=5
            )
            processing_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                output = result["enhanced_description"]
                narrative_quality = self.assess_narrative_quality(output)
                semantic_accuracy = self.calculate_semantic_accuracy(output, self.ground_truths[scene_key])
                
                return {
                    "description": output,
                    "time_ms": processing_time,
                    "narrative_quality": narrative_quality,
                    "semantic_accuracy": semantic_accuracy,
                    "success": True
                }
        except Exception as e:
            print(f"Our system error: {e}")
        
        return {"success": False, "time_ms": 0, "description": "", "narrative_quality": 0.0, "semantic_accuracy": 0.0}
    
    def test_claude_opus(self, scene_description, scene_key):
        """Test Claude Opus with real API"""
        start_time = time.time()
        try:
            response = self.claude_client.messages.create(
                model=self.claude_opus_id,
                max_tokens=100,
                messages=[{
                    "role": "user", 
                    "content": f"Describe this scene vividly in one cinematic sentence: {scene_description}"
                }]
            )
            processing_time = (time.time() - start_time) * 1000
            
            description = response.content[0].text
            narrative_quality = self.assess_narrative_quality(description)
            semantic_accuracy = self.calculate_semantic_accuracy(description, self.ground_truths[scene_key])
            
            return {
                "description": description,
                "time_ms": processing_time,
                "narrative_quality": narrative_quality,
                "semantic_accuracy": semantic_accuracy,
                "success": True
            }
        except Exception as e:
            print(f"Claude error: {e}")
            return {"success": False, "time_ms": 0, "description": "", "narrative_quality": 0.0, "semantic_accuracy": 0.0}
    
    def test_gpt4_turbo(self, scene_description, scene_key):
        """Test GPT-4 Turbo with real API"""
        start_time = time.time()
        try:
            response = self.openai_client.chat.completions.create(
                model=self.gpt4_turbo_id,
                messages=[{
                    "role": "user",
                    "content": f"Describe this scene vividly in one cinematic sentence: {scene_description}"
                }],
                max_tokens=100
            )
            processing_time = (time.time() - start_time) * 1000
            
            description = response.choices[0].message.content
            narrative_quality = self.assess_narrative_quality(description)
            semantic_accuracy = self.calculate_semantic_accuracy(description, self.ground_truths[scene_key])
            
            return {
                "description": description,
                "time_ms": processing_time,
                "narrative_quality": narrative_quality,
                "semantic_accuracy": semantic_accuracy,
                "success": True
            }
        except Exception as e:
            print(f"GPT-4 error: {e}")
            return {"success": False, "time_ms": 0, "description": "", "narrative_quality": 0.0, "semantic_accuracy": 0.0}
    
    def run_benchmark(self):
        """Run benchmark with real API calls and real scores"""
        test_scenes = [
            ("A car driving through a city at night with neon lights", "car_city"),
            ("A person dancing in a room with colorful lighting effects", "person_dancing")
        ]
        
        print("🚀 REAL SCORES BENCHMARK - ACTUAL API PERFORMANCE")
        print("=" * 70)
        print("🎯 Testing with Real Claude & OpenAI APIs")
        print("=" * 70)
        
        our_results = []
        claude_results = []
        gpt4_results = []
        
        for scene, scene_key in test_scenes:
            print(f"\n🎬 TEST SCENE: {scene}")
            print("-" * 50)
            
            # Test our system
            our_result = self.test_our_system(scene, scene_key)
            if our_result["success"]:
                our_results.append(our_result)
                print(f"✅ VISUAL NARRATOR:")
                print(f"   ⚡ {our_result['time_ms']:.1f}ms")
                print(f"   🎬 Narrative: {our_result['narrative_quality']:.1%}")
                print(f"   🎯 Semantic: {our_result['semantic_accuracy']:.1%}")
                print(f"   💎 '{our_result['description']}'")
            
            # Test Claude Opus
            claude_result = self.test_claude_opus(scene, scene_key)
            if claude_result["success"]:
                claude_results.append(claude_result)
                print(f"✅ CLAUDE OPUS:")
                print(f"   ⚡ {claude_result['time_ms']:.1f}ms") 
                print(f"   🎬 Narrative: {claude_result['narrative_quality']:.1%}")
                print(f"   🎯 Semantic: {claude_result['semantic_accuracy']:.1%}")
                desc = claude_result['description']
                if len(desc) > 70:
                    desc = desc[:67] + "..."
                print(f"   💎 '{desc}'")
            
            # Test GPT-4 Turbo
            gpt4_result = self.test_gpt4_turbo(scene, scene_key)
            if gpt4_result["success"]:
                gpt4_results.append(gpt4_result)
                print(f"✅ GPT-4 TURBO:")
                print(f"   ⚡ {gpt4_result['time_ms']:.1f}ms")
                print(f"   🎬 Narrative: {gpt4_result['narrative_quality']:.1%}")
                print(f"   🎯 Semantic: {gpt4_result['semantic_accuracy']:.1%}")
                desc = gpt4_result['description']
                if len(desc) > 70:
                    desc = desc[:67] + "..."
                print(f"   💎 '{desc}'")
        
        # Generate summary with real calculated scores
        if our_results and claude_results and gpt4_results:
            self.print_real_summary(our_results, claude_results, gpt4_results)
        else:
            print("\n⚠️  Some API calls failed - check connectivity and API keys")
    
    def print_real_summary(self, our_results, claude_results, gpt4_results):
        """Print summary with real calculated scores"""
        print("\n" + "=" * 70)
        print("🏆 COMPETITIVE POSITIONING - REAL SCORES")
        print("=" * 70)
        
        # Calculate real averages from API responses
        our_time = statistics.mean([r["time_ms"] for r in our_results])
        claude_time = statistics.mean([r["time_ms"] for r in claude_results])
        gpt4_time = statistics.mean([r["time_ms"] for r in gpt4_results])
        
        our_narrative = statistics.mean([r["narrative_quality"] for r in our_results])
        claude_narrative = statistics.mean([r["narrative_quality"] for r in claude_results])
        gpt4_narrative = statistics.mean([r["narrative_quality"] for r in gpt4_results])
        
        our_semantic = statistics.mean([r["semantic_accuracy"] for r in our_results])
        claude_semantic = statistics.mean([r["semantic_accuracy"] for r in claude_results])
        gpt4_semantic = statistics.mean([r["semantic_accuracy"] for r in gpt4_results])
        
        print(f"\n📊 REAL PERFORMANCE METRICS")
        print(f"• Visual Narrator: {our_time:.1f}ms | {our_narrative:.1%} Narrative | {our_semantic:.1%} Semantic")
        print(f"• Claude Opus:     {claude_time:.1f}ms | {claude_narrative:.1%} Narrative | {claude_semantic:.1%} Semantic") 
        print(f"• GPT-4 Turbo:     {gpt4_time:.1f}ms | {gpt4_narrative:.1%} Narrative | {gpt4_semantic:.1%} Semantic")
        
        print(f"\n🎯 COMPETITIVE ADVANTAGES")
        print(f"✅ SPEED: {claude_time/our_time:.0f}x faster than Claude Opus")
        print(f"✅ QUALITY: {our_semantic/claude_semantic*100:.0f}% of Claude's semantic accuracy") 
        print(f"✅ NARRATIVE: {our_narrative:.1%} professional quality")
        print(f"✅ COST: Zero marginal cost vs. API pricing")
        
        print(f"\n💎 SAMPLE OUTPUTS (Real API Results)")
        for i, result in enumerate(our_results, 1):
            print(f"{i}. {result['description']}")
        
        print(f"\n📈 METHODOLOGY")
        print("• Narrative Quality: 8-point checklist (capitalization, punctuation, flow, etc.)")
        print("• Semantic Accuracy: Cosine similarity to ground truth descriptions")
        print("• Speed: Actual API response times measured")
        print("• All scores calculated from real API responses")

if __name__ == "__main__":
    print("🔧 Initializing Real Scores Benchmark...")
    benchmark = RealScoresBenchmark()
    benchmark.run_benchmark()
