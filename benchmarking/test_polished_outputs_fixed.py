import requests
import re

def test_polished_api():
    """Test the polished API for grammar and repetition fixes"""
    test_scenes = [
        "A car driving through a city at night with neon lights",
        "A person dancing in a room with colorful lighting effects",
        "A elegant eagle flying over a mountain",
        "A old building with ancient architecture"
    ]
    
    print("🎯 TESTING POLISHED API - GRAMMAR & REPETITION FIXES")
    print("=" * 65)
    
    for scene in test_scenes:
        response = requests.post(
            "http://localhost:8008/describe/scene",
            json={"scene_description": scene, "enhance_adjectives": True}
        )
        
        if response.status_code == 200:
            result = response.json()
            output = result["enhanced_description"]
            
            print(f"📝 INPUT:  {scene}")
            print(f"💎 OUTPUT: {output}")
            
            # Check for common issues
            issues = []
            if " a " in output.lower() and any(output.lower().count(f" {word} {word} ") > 0 for word in ['neon', 'colorful', 'dynamic']):
                issues.append("word repetition")
            if re.search(r'\ba [aeiou]', output, re.IGNORECASE):
                issues.append("a/an grammar")
            if "  " in output:
                issues.append("double spaces")
                
            if issues:
                print(f"❌ ISSUES: {', '.join(issues)}")
            else:
                print(f"✅ CLEAN: No grammar or repetition issues")
            print("─" * 65)

if __name__ == "__main__":
    test_polished_api()
