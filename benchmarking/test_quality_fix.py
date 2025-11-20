import requests
import json
import time

def test_quality_fixed_api():
    """Test the new quality-fixed API"""
    test_scenes = [
        "A car driving through a city at night with neon lights",
        "A person dancing in a room with colorful lighting effects",
        "A mountain landscape with sunset and trees"
    ]
    
    print("🧪 TESTING QUALITY-FIXED API OUTPUTS...")
    print("=" * 60)
    
    for scene in test_scenes:
        try:
            response = requests.post(
                "http://localhost:8003/describe/scene",
                json={
                    "scene_description": scene,
                    "enhance_adjectives": True,
                    "adjective_density": 0.6,
                    "grammar_constraint": True
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"🎯 INPUT:  {scene}")
                print(f"🎨 OUTPUT: {result['enhanced_description']}")
                print(f"⏱️  TIME:   {result['processing_time']:.2f}ms")
                print("─" * 50)
            else:
                print(f"❌ Failed for: {scene}")
                
        except Exception as e:
            print(f"💥 Error: {e}")

if __name__ == "__main__":
    test_quality_fixed_api()
