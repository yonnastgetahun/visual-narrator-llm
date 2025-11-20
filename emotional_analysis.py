import json

class EmotionalAnalyzer:
    def __init__(self):
        with open('emotional_scenes_dataset.json', 'r') as f:
            self.scenes = json.load(f)
    
    def analyze_scene_emotion(self, description):
        """Analyze emotional content of a scene description"""
        description_lower = description.lower()
        
        # Simple keyword-based emotional analysis
        emotional_indicators = {
            'action': ['chase', 'explosion', 'fight', 'escape', 'battle'],
            'drama': ['romantic', 'emotional', 'confession', 'tense', 'verdict'],
            'comedy': ['comedic', 'awkward', 'prank', 'funny', 'slip'],
            'horror': ['haunted', 'scare', 'dark', 'terror', 'fear'],
            'documentary': ['nature', 'wildlife', 'historical', 'educational']
        }
        
        scores = {emotion: 0 for emotion in emotional_indicators}
        
        for emotion, keywords in emotional_indicators.items():
            for keyword in keywords:
                if keyword in description_lower:
                    scores[emotion] += 1
        
        # Normalize scores
        total = sum(scores.values()) or 1
        normalized_scores = {k: v/total for k, v in scores.items()}
        
        # Get dominant emotion
        dominant_emotion = max(normalized_scores, key=normalized_scores.get)
        
        return {
            'dominant_emotion': dominant_emotion,
            'emotional_scores': normalized_scores,
            'confidence': normalized_scores[dominant_emotion]
        }

# Test the analyzer
analyzer = EmotionalAnalyzer()
test_scenes = [
    "a car chase through the city with explosions",
    "a romantic walk on the beach at sunset", 
    "a funny slip on a banana peel"
]

print("Emotional Analysis Test:")
for scene in test_scenes:
    analysis = analyzer.analyze_scene_emotion(scene)
    print(f"\nScene: '{scene}'")
    print(f"Dominant emotion: {analysis['dominant_emotion']}")
    print(f"Confidence: {analysis['confidence']:.2f}")
