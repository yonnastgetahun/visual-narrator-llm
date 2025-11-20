import requests
import time

def test_proper_api():
    """Test the proper API with comprehensive evaluation"""
    test_scenes = [
        "A car driving through a city at night with neon lights",
        "A person dancing in a room with colorful lighting effects",
        "A mountain landscape with sunset and trees",
        "A modern building with glass windows reflecting sunlight"
    ]
    
    print("🧪 TESTING PROPER API - COMPREHENSIVE EVALUATION")
    print("=" * 65)
    
    for scene in test_scenes:
        try:
            start_time = time.time()
            response = requests.post(
                "http://localhost:8006/describe/scene",
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
                
                # Quality metrics
                words = output.split()
                adjective_count = self.count_adjectives(output)
                sentence_quality = self.assess_sentence_quality(output)
                
                print(f"📊 METRICS: {len(words)} words, {adjective_count} adjectives")
                print(f"🎯 QUALITY: {sentence_quality}")
                print("─" * 65)
                
            else:
                print(f"❌ FAILED: {scene}")
                print("─" * 65)
                
        except Exception as e:
            print(f"💥 ERROR: {e}")
            print("─" * 65)
    
    def count_adjectives(self, text):
        """Count quality adjectives in text"""
        quality_adjectives = [
            'sleek', 'modern', 'gleaming', 'luxurious', 'sporty', 'vibrant',
            'bustling', 'illuminated', 'colorful', 'glowing', 'dazzling',
            'energetic', 'graceful', 'expressive', 'charismatic', 'dynamic',
            'atmospheric', 'majestic', 'towering', 'snow-capped', 'rugged',
            'breathtaking', 'dramatic', 'picturesque', 'stunning', 'lush',
            'verdant', 'imposing', 'architectural', 'reflective', 'shimmering',
            'golden', 'warm', 'brilliant', 'radiant'
        ]
        text_lower = text.lower()
        return sum(1 for adj in quality_adjectives if adj in text_lower)
    
    def assess_sentence_quality(self, text):
        """Assess basic sentence quality"""
        if not text:
            return "Poor: Empty output"
        
        # Check for proper sentence structure
        has_capital = text[0].isupper() if text else False
        has_period = text.endswith('.') if text else False
        word_count = len(text.split())
        
        # Check for common issues
        issues = []
        if not has_capital:
            issues.append("no capitalization")
        if not has_period:
            issues.append("no ending punctuation")
        if word_count < 3:
            issues.append("too short")
        if word_count > 25:
            issues.append("too long")
        if ' .' in text or ' ,' in text:
            issues.append("spacing before punctuation")
        
        if not issues:
            return "Excellent: Proper structure"
        elif len(issues) == 1:
            return f"Good: Minor issue ({issues[0]})"
        else:
            return f"Needs work: {', '.join(issues)}"

if __name__ == "__main__":
    test_proper_api()
