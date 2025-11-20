#!/usr/bin/env python3
"""
PROPER Visual Narrator VLM API
- Correct sentence structure
- Consistent adjective application
- Natural language flow
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import random
import re

app = FastAPI(title="Visual Narrator VLM API - PROPER", version="4.1.0")

class SceneRequest(BaseModel):
    scene_description: str
    enhance_adjectives: bool = True

class ProperVisualSystem:
    def __init__(self):
        self.adjective_map = {
            "car": ["sleek", "modern", "gleaming", "luxurious", "sporty"],
            "driving": ["smoothly", "gracefully", "effortlessly"],
            "city": ["vibrant", "bustling", "modern", "illuminated"],
            "night": ["dark", "starry", "moonlit", "clear"],
            "neon": ["colorful", "vibrant", "glowing", "dazzling", "radiant"],
            "lights": ["bright", "colorful", "dazzling", "glowing"],
            "person": ["energetic", "graceful", "expressive", "charismatic"],
            "dancing": ["rhythmically", "gracefully", "energetically", "expressively"],
            "room": ["dimly-lit", "colorful", "dynamic", "atmospheric"],
            "lighting": ["dynamic", "colorful", "vibrant", "atmospheric"],
            "effects": ["dramatic", "captivating", "mesmerizing", "dynamic"],
            "mountain": ["majestic", "towering", "snow-capped", "rugged"],
            "landscape": ["breathtaking", "dramatic", "picturesque", "stunning"],
            "sunset": ["vibrant", "golden", "dramatic", "colorful"],
            "trees": ["lush", "towering", "verdant", "majestic"],
            "building": ["modern", "imposing", "architectural", "contemporary"],
            "glass": ["reflective", "shimmering", "clear", "transparent"],
            "windows": ["gleaming", "reflective", "sparkling", "clear"],
            "sunlight": ["golden", "warm", "brilliant", "radiant"]
        }
    
    def enhance_description(self, description):
        """Properly enhance description with natural adjectives"""
        words = description.lower().split()
        enhanced_words = []
        
        i = 0
        while i < len(words):
            word = words[i]
            clean_word = re.sub(r'[^a-z]', '', word)
            
            # Find matching adjectives for this word
            added_adjectives = []
            for category, adjectives in self.adjective_map.items():
                if (clean_word == category or 
                    clean_word in category or 
                    category in clean_word or
                    self.get_word_root(clean_word) == self.get_word_root(category)):
                    
                    if adjectives and random.random() < 0.7:  # 70% chance to add adjectives
                        num_adjs = random.randint(1, 2)
                        selected = random.sample(adjectives, min(num_adjs, len(adjectives)))
                        added_adjectives.extend(selected)
            
            # Add adjectives before the word (remove duplicates)
            added_adjectives = list(dict.fromkeys(added_adjectives))[:2]  # Max 2 adjectives
            enhanced_words.extend(added_adjectives)
            enhanced_words.append(word)
            i += 1
        
        # Build proper sentences
        return self.build_proper_sentence(enhanced_words, description)
    
    def get_word_root(self, word):
        """Simple word root extraction"""
        if word.endswith('ing'):
            return word[:-3]
        elif word.endswith('s'):
            return word[:-1]
        return word
    
    def build_proper_sentence(self, words, original):
        """Build grammatically correct sentences"""
        if not words:
            return original.capitalize()
        
        # Join words and fix basic grammar
        text = ' '.join(words)
        
        # Capitalize first letter
        text = text[0].upper() + text[1:] if text else text
        
        # Ensure it ends with a period
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        # Fix common issues
        text = re.sub(r'\s+([,.!?])', r'\1', text)  # Remove spaces before punctuation
        text = re.sub(r'\.\s*\.', '.', text)  # Remove double periods
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        
        # Ensure reasonable length - don't create run-on sentences
        if len(text.split()) > 15:
            # Split into two sentences at a natural point
            words_in_text = text.split()
            split_point = len(words_in_text) // 2
            
            # Find a good split point (after a noun)
            nouns = ['car', 'city', 'night', 'lights', 'person', 'room', 
                    'mountain', 'sunset', 'trees', 'building', 'windows']
            
            for i in range(split_point, len(words_in_text)):
                if words_in_text[i].strip('.,').lower() in nouns:
                    split_point = i + 1
                    break
            
            first_part = ' '.join(words_in_text[:split_point])
            second_part = ' '.join(words_in_text[split_point:])
            
            # Ensure both parts are properly formatted
            if not first_part.endswith('.'):
                first_part += '.'
            if second_part and not second_part[0].isupper():
                second_part = second_part[0].upper() + second_part[1:]
            if second_part and not second_part.endswith('.'):
                second_part += '.'
            
            text = first_part + ' ' + second_part
        
        return text

proper_system = ProperVisualSystem()

@app.post("/describe/scene")
async def describe_scene(request: SceneRequest):
    """Generate properly formatted descriptions"""
    try:
        start_time = time.time()
        
        if request.enhance_adjectives:
            enhanced_desc = proper_system.enhance_description(request.scene_description)
        else:
            enhanced_desc = request.scene_description
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "enhanced_description": enhanced_desc,
            "original_description": request.scene_description,
            "processing_time": processing_time,
            "version": "4.1.0-proper"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Description generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 STARTING PROPER VISUAL NARRATOR API...")
    print("📍 API: http://localhost:8006")
    print("🎯 FOCUS: Grammatically correct, naturally flowing descriptions")
    uvicorn.run(app, host="0.0.0.0", port=8006)
