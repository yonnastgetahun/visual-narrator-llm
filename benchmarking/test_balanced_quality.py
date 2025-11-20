import requests
import json
import time

def test_balanced_api():
    """Test the balanced quality API"""
    test_scenes = [
        "A car driving through a city at night with neon lights",
        "A person dancing in a room with colorful lighting effects", 
        "A mountain landscape with sunset and trees",
        "A modern building with glass windows reflecting sunlight"
    ]
    
    print("🎯 TESTING BALANCED QUALITY API")
    print("=" * 60)
    
    for scene in test_scenes:
        try:
            response = requests.post(
                "http://localhost:8004/describe/scene",
                json={
                    "scene_description": scene,
                    "enhance_adjectives": True,
                    "adjective_density": 0.8,  # Balanced density
                    "grammar_constraint": True
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"📝 INPUT:  {scene}")
                print(f"🎨 OUTPUT: {result['enhanced_description']}")
                print(f"⚡ TIME:   {result['processing_time']:.2f}ms")
                
                # Calculate adjective count
                adjectives = ['sleek', 'gleaming', 'modern', 'vibrant', 'colorful', 
                            'energetic', 'graceful', 'majestic', 'dramatic', 'lush']
                output = result['enhanced_description'].lower()
                adj_count = sum(1 for adj in adjectives if adj in output)
                print(f"📊 ADJECTIVES: {adj_count} (Target: 2-4)")
                print("─" * 50)
            else:
                print(f"❌ Failed for: {scene}")
                
        except Exception as e:
            print(f"💥 Error: {e}")

if __name__ == "__main__":
    test_balanced_api()
