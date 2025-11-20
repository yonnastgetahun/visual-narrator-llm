import requests
import re

def test_clean_api():
    """Test the clean API for natural, flowing outputs"""
    test_scenes = [
        "A car driving through a city at night with neon lights",
        "A person dancing in a room with colorful lighting effects",
        "A elegant eagle flying over a mountain",
        "A old building with ancient architecture"
    ]
    
    print("🎯 TESTING CLEAN API - NATURAL FLOWING OUTPUTS")
    print("=" * 65)
    
    for scene in test_scenes:
        response = requests.post(
            "http://localhost:8010/describe/scene",
            json={"scene_description": scene, "enhance_adjectives": True}
        )
        
        if response.status_code == 200:
            result = response.json()
            output = result["enhanced_description"]
            
            print(f"📝 INPUT:  {scene}")
            print(f"💎 OUTPUT: {output}")
            
            # Check for quality
            issues = self.assess_output_quality(output)
            if issues:
                print(f"❌ ISSUES: {', '.join(issues)}")
            else:
                print(f"✅ CLEAN: Natural, flowing description")
            print("─" * 65)
    
    def assess_output_quality(self, text):
        """Assess output for common issues"""
        issues = []
        
        # Check for awkward phrases
        awkward_phrases = ['reflecting on atmosphere', 'modern streamlined a', 'neon neon', 'colorful colorful']
        if any(phrase in text.lower() for phrase in awkward_phrases):
            issues.append("awkward phrasing")
        
        # Check grammar
        if re.search(r'\ba [aeiou]', text, re.IGNORECASE):
            issues.append("a/an grammar")
        
        # Check repetition
        words = text.lower().split()
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                issues.append("word repetition")
                break
        
        # Check sentence structure
        if not text[0].isupper():
            issues.append("capitalization")
        if not text.endswith('.'):
            issues.append("punctuation")
        
        return issues

if __name__ == "__main__":
    test_clean_api()
