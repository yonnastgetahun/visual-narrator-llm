import requests

def validate_quality():
    """Compare our outputs against Claude-like quality"""
    
    # Claude's outputs from original benchmark
    claude_outputs = {
        "car_city": "As the sleek, midnight-black car glides through the city streets, the urban landscape comes alive with a dazzling display of neon lights.",
        "person_dancing": "In the dimly lit room, a lone figure stands in the center, their silhouette illuminated by an array of vibrant, pulsating lights."
    }
    
    our_scenes = [
        "A car driving through a city at night with neon lights",
        "A person dancing in a room with colorful lighting effects"
    ]
    
    print("🔍 QUALITY VALIDATION - VS CLAUDE STANDARD")
    print("=" * 65)
    
    for scene in our_scenes:
        response = requests.post(
            "http://localhost:8006/describe/scene",
            json={"scene_description": scene, "enhance_adjectives": True}
        )
        
        if response.status_code == 200:
            result = response.json()
            our_output = result["enhanced_description"]
            
            print(f"\n🎯 SCENE: {scene}")
            print(f"💎 OUR OUTPUT:    {our_output}")
            
            # Find matching Claude output
            claude_key = "car_city" if "car" in scene else "person_dancing"
            claude_output = claude_outputs[claude_key]
            print(f"🏆 CLAUDE OUTPUT: {claude_output}")
            
            # Basic comparison
            our_words = len(our_output.split())
            claude_words = len(claude_output.split())
            our_adjectives = sum(1 for word in our_output.split() if any(adj in word for adj in ['sleek', 'vibrant', 'dazzling', 'modern', 'colorful']))
            
            print(f"📊 COMPARISON: {our_words} words, {our_adjectives} adjectives vs Claude's {claude_words} words")
            print("─" * 65)

if __name__ == "__main__":
    validate_quality()
