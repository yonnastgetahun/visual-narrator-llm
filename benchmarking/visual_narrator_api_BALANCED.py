#!/usr/bin/env python3
"""
BALANCED Visual Narrator VLM API
- Natural language flow WITH rich descriptions
- Balanced adjective density (2.0 target)
- Proper grammar and sentence structure
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time
import random
import re
from datetime import datetime

app = FastAPI(title="Visual Narrator VLM API - BALANCED", version="3.2.0")

class SceneDescriptionRequest(BaseModel):
    scene_description: str
    enhance_adjectives: bool = True
    include_spatial: bool = True  
    adjective_density: float = 0.8  # BALANCED - between 0.6 and 1.0
    grammar_constraint: bool = True

class BalancedVisualSystem:
    def __init__(self):
        self.adjective_library = self.create_balanced_adjective_library()
        self.spatial_terms = ["left", "right", "above", "below", "behind", "in front of", 
                             "near", "beside", "next to", "between", "under", "over"]
    
    def create_balanced_adjective_library(self):
        """Curated adjectives for natural but rich descriptions"""
        return {
            "car": ["sleek", "gleaming", "modern", "powerful", "luxurious", "sporty"],
            "person": ["energetic", "graceful", "expressive", "charismatic", "animated"],
            "city": ["vibrant", "bustling", "modern", "illuminated", "lively"],
            "night": ["dark", "starry", "moonlit", "inky", "nocturnal"],
            "lights": ["neon", "colorful", "vibrant", "dazzling", "glowing", "radiant"],
            "dancing": ["energetic", "graceful", "rhythmic", "expressive", "dynamic"],
            "room": ["dimly-lit", "colorful", "dynamic", "atmospheric", "vibrant"],
            "mountain": ["majestic", "towering", "snow-capped", "rugged", "imposing"],
            "sunset": ["vibrant", "golden", "dramatic", "colorful", "breathtaking"],
            "trees": ["lush", "towering", "verdant", "ancient", "majestic"]
        }
    
    def enhance_with_balanced_constraints(self, description, density=0.8):
        """Balanced adjective injection with natural flow"""
        words = description.split()
        enhanced_words = []
        
        i = 0
        while i < len(words):
            word = words[i]
            clean_word = word.strip('.,!?;:').lower()
            
            # Check if this word or related words have adjectives
            added_adjectives = False
            
            for category, adjectives in self.adjective_library.items():
                if category in clean_word or self.is_related(clean_word, category):
                    # Add 1-2 quality adjectives (not 4+, not 0)
                    if random.random() < density and len(adjectives) > 0:
                        num_adjs = random.randint(1, 2)  # 1-2 adjectives max
                        selected_adjs = random.sample(adjectives, min(num_adjs, len(adjectives)))
                        
                        # Insert before the noun
                        enhanced_words.extend(selected_adjs)
                        added_adjectives = True
            
            enhanced_words.append(word)
            i += 1
        
        # Build natural sentences
        raw_output = " ".join(enhanced_words)
        return self.build_natural_sentences(raw_output, description)
    
    def is_related(self, word, category):
        """Check if word is related to category"""
        relations = {
            "car": ["vehicle", "automobile", "sedan", "sports car"],
            "city": ["urban", "metropolis", "downtown", "streets"],
            "lights": ["lighting", "illumination", "glow", "neon"],
            "dancing": ["dancer", "dances", "movement"],
            "room": ["space", "area", "chamber", "environment"],
            "mountain": ["peak", "summit", "range", "hill"],
            "sunset": ["dusk", "twilight", "evening", "sundown"],
            "trees": ["forest", "woodland", "foliage", "pines"]
        }
        
        if category in relations:
            return word in relations[category]
        return False
    
    def build_natural_sentences(self, enhanced_text, original_description):
        """Build natural, flowing sentences from enhanced words"""
        words = enhanced_text.split()
        
        if len(words) <= 8:
            # Simple sentence - just capitalize and add period
            sentence = " ".join(words)
            return sentence.capitalize() + "."
        
        # For longer descriptions, create proper sentences
        sentences = []
        current_sentence = []
        
        for word in words:
            current_sentence.append(word)
            
            # End sentence at natural points (after 6-12 words)
            if (len(current_sentence) >= 8 and 
                random.random() < 0.3 and 
                word not in ['with', 'and', 'or', 'the']):
                
                sentence = " ".join(current_sentence)
                sentences.append(sentence.capitalize())
                current_sentence = []
        
        # Add any remaining words
        if current_sentence:
            sentence = " ".join(current_sentence)
            sentences.append(sentence.capitalize())
        
        # Join sentences properly
        if sentences:
            result = ". ".join(sentences)
            if not result.endswith('.'):
                result += '.'
            return result
        else:
            return original_description.capitalize() + "."

# Initialize balanced system
balanced_system = BalancedVisualSystem()

@app.post("/describe/scene")
async def describe_scene(request: SceneDescriptionRequest):
    """Generate balanced quality descriptions"""
    try:
        start_time = time.time()
        
        if request.enhance_adjectives:
            enhanced_desc = balanced_system.enhance_with_balanced_constraints(
                request.scene_description, 
                request.adjective_density
            )
        else:
            enhanced_desc = request.scene_description
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "enhanced_description": enhanced_desc,
            "original_description": request.scene_description,
            "processing_time": processing_time,
            "version": "3.2.0-balanced"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Description generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 STARTING BALANCED VISUAL NARRATOR API...")
    print("📍 API: http://localhost:8004")
    print("🎯 FOCUS: Rich but natural descriptions")
    uvicorn.run(app, host="0.0.0.0", port=8004)
