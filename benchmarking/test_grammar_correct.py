import requests
import time

def test_grammar_correct():
    """Test the grammar-correct API"""
    test_scenes = [
        "A car driving through a city at night with neon lights",
        "A person dancing in a room with colorful lighting effects",
        "A mountain landscape with sunset and trees",
        "A modern building with glass windows reflecting sunlight"
    ]
    
    print("🎯 TESTING GRAMMAR-CORRECT API")
    print("=" * 65)
    
    for scene in test_scenes:
        try:
            start_time = time.time()
            response = requests.post(
                "http://localhost:8007/describe/scene",
                json={
                    "scene_description": scene,
                    "enhance_adjectives": True
                },
                timeout=10
            )
            processing_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                output = result["enhanced_description"]
                
                print(f"📝 INPUT:  {scene}")
                print(f"💎 OUTPUT: {output}")
                print(f"⚡ TIME:   {processing_time:.2f}ms")
                
                # Quality assessment
                quality = assess_quality(output)
                print(f"🎯 QUALITY: {quality}")
                print("─" * 65)
                
            else:
                print(f"❌ FAILED: {scene}")
                print("─" * 65)
                
        except Exception as e:
            print(f"💥 ERROR: {e}")
            print("─" * 65)

def assess_quality(text):
    """Assess output quality"""
    if not text:
        return "Poor"
    
    # Check for proper grammar indicators
    checks = [
        text[0].isupper(),  # Starts with capital
        text.endswith('.'),  # Ends with period
        ' a ' not in text.lower() or ' a ' in text.lower() and text.lower().count(' a ') == 1,  # Proper article usage
        len(text.split()) >= 8 and len(text.split()) <= 25,  # Reasonable length
        not any(word in text.lower() for word in [' a a ', ' the the ', '  '])  # No obvious repeats
    ]
    
    score = sum(checks)
    
    if score >= 4:
        return "Excellent ✓"
    elif score >= 3:
        return "Good ✓" 
    else:
        return "Needs work"

if __name__ == "__main__":
    test_grammar_correct()
