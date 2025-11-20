import requests
import json
import re

class GrammarQualityFix:
    """Fix the keyword stuffing issue while maintaining adjective richness"""
    
    def __init__(self):
        self.api_url = "http://localhost:8002"
    
    def enhance_with_grammar_constraints(self, description, max_adjectives_per_noun=2):
        """Enhanced adjective injection with grammar constraints"""
        try:
            response = requests.post(
                f"{self.api_url}/describe/scene",
                json={
                    "scene_description": description,
                    "enhance_adjectives": True,
                    "include_spatial": True,
                    "adjective_density": 0.6,  # REDUCED from 1.0 to prevent stuffing
                    "grammar_constraint": True  # New parameter we need to implement
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                raw_output = result["enhanced_description"]
                
                # Apply post-processing grammar fixes
                cleaned_output = self.fix_broken_grammar(raw_output)
                return cleaned_output
                
        except Exception as e:
            print(f"API Error: {e}")
        
        return description
    
    def fix_broken_grammar(self, text):
        """Fix common grammar issues from adjective stuffing"""
        # Fix "adjective adjective adjective A noun" pattern
        text = re.sub(r'(\w+ ){3,}A? (\w+)', self._fix_adjective_clusters, text)
        
        # Ensure sentence starts with proper capitalization
        sentences = text.split('. ')
        fixed_sentences = []
        
        for sentence in sentences:
            if sentence.strip():
                # Capitalize first letter
                sentence = sentence[0].upper() + sentence[1:] if sentence else sentence
                fixed_sentences.append(sentence)
        
        return '. '.join(fixed_sentences)
    
    def _fix_adjective_clusters(self, match):
        """Fix clusters of adjectives before nouns"""
        words = match.group(0).split()
        nouns = ['car', 'person', 'building', 'tree', 'water', 'sky', 'mountain']
        
        # Find the first noun
        noun_index = None
        for i, word in enumerate(words):
            if word.lower() in nouns or (word.lower() == 'a' and i+1 < len(words) and words[i+1].lower() in nouns):
                noun_index = i
                break
        
        if noun_index and noun_index > 0:
            # Keep only 2 adjectives before noun
            adjectives = words[:noun_index][-2:]  # Last 2 adjectives only
            rest = words[noun_index:]
            return ' '.join(adjectives + rest)
        
        return match.group(0)

# Test the fix
fixer = GrammarQualityFix()

test_scenes = [
    "A car driving through a city at night with neon lights",
    "A person dancing in a room with colorful lighting effects"
]

print("🔧 TESTING GRAMMAR FIXES...")
for scene in test_scenes:
    fixed_output = fixer.enhance_with_grammar_constraints(scene)
    print(f"Input: {scene}")
    print(f"Fixed: {fixed_output}")
    print("---")
