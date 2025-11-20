import requests

def final_quality_comparison():
    """Final quality comparison against Claude standard"""
    
    # Claude's gold-standard outputs
    claude_standards = {
        "car": "As the sleek, midnight-black car glides through the city streets, the urban landscape comes alive with a dazzling display of neon lights.",
        "dancing": "In the dimly lit room, a lone figure stands in the center, their silhouette illuminated by an array of vibrant, pulsating lights."
    }
    
    test_scenes = [
        "A car driving through a city at night with neon lights",
        "A person dancing in a room with colorful lighting effects"
    ]
    
    print("🏆 FINAL QUALITY COMPARISON")
    print("=" * 70)
    
    for i, scene in enumerate(test_scenes):
        response = requests.post(
            "http://localhost:8007/describe/scene",
            json={"scene_description": scene, "enhance_adjectives": True}
        )
        
        if response.status_code == 200:
            result = response.json()
            our_output = result["enhanced_description"]
            claude_output = claude_standards["car"] if i == 0 else claude_standards["dancing"]
            
            print(f"\n🎯 SCENE: {scene}")
            print(f"💎 OUR OUTPUT:")
            print(f"   {our_output}")
            print(f"🏆 CLAUDE OUTPUT:")
            print(f"   {claude_output}")
            
            # Comparative analysis
            our_words = len(our_output.split())
            claude_words = len(claude_output.split())
            our_adj = count_quality_adjectives(our_output)
            claude_adj = count_quality_adjectives(claude_output)
            
            print(f"📊 ANALYSIS: {our_words} words, {our_adj} adjectives vs {claude_words} words, {claude_adj} adjectives")
            print("─" * 70)

def count_quality_adjectives(text):
    """Count quality adjectives"""
    quality_adj = [
        'sleek', 'modern', 'gleaming', 'luxurious', 'vibrant', 'bustling',
        'illuminated', 'colorful', 'glowing', 'dazzling', 'energetic',
        'graceful', 'expressive', 'charismatic', 'dynamic', 'atmospheric',
        'majestic', 'towering', 'dramatic', 'lush', 'verdant', 'imposing',
        'reflective', 'golden', 'radiant', 'midnight-black', 'urban',
        'dazzling', 'vibrant', 'pulsating'
    ]
    text_lower = text.lower()
    return sum(1 for adj in quality_adj if adj in text_lower)

if __name__ == "__main__":
    final_quality_comparison()
