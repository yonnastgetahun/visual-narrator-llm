#!/usr/bin/env python3
"""
QUALITY-FIXED Visual Narrator VLM API
- Addresses "keyword stuffing" issue
- Maintains adjective richness with natural flow
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time
import random
import re
from datetime import datetime

app = FastAPI(title="Visual Narrator VLM API - QUALITY FIXED", version="3.1.0")

class SceneDescriptionRequest(BaseModel):
    scene_description: str
    enhance_adjectives: bool = True
    include_spatial: bool = True  
    adjective_density: float = 0.6  # REDUCED from 1.0 to prevent stuffing
    grammar_constraint: bool = True  # NEW: Enable grammar fixes

class QualityFixedVisualSystem:
    def __init__(self):
        self.adjective_library = self.create_quality_adjective_library()
        self.spatial_terms = ["left", "right", "above", "below", "behind", "in front of", 
                             "near", "beside", "next to", "between", "under", "over"]
    
    def create_quality_adjective_library(self):
        """Curated adjectives that work well in natural language"""
        return {
            "person": ["expressive", "graceful", "energetic", "charismatic"],
            "car": ["sleek", "gleaming", "modern", "powerful"],
            "building": ["majestic", "modern", "imposing", "architectural"],
            "tree": ["lush", "towering", "ancient", "verdant"],
            "mountain": ["majestic", "snow-capped", "rugged", "towering"],
            "water": ["serene", "glistening", "tranquil", "sparkling"],
            "sky": ["dramatic", "expansive", "vibrant", "colorful"],
            "sunset": ["stunning", "vibrant", "dramatic", "golden"]
        }
    
    def enhance_with_quality_constraints(self, description, density=0.6):
        """Enhanced adjective injection with quality constraints"""
        words = description.split()
        enhanced_words = []
        adjective_count = 0
        max_adjectives = int(density * 8)  # REDUCED maximum
        
        i = 0
        while i < len(words):
            word = words[i]
            clean_word = word.strip('.,!?;:').lower()
            
            # Add QUALITY adjectives (not quantity)
            if clean_word in self.adjective_library and adjective_count < max_adjectives:
                adjectives = self.adjective_library[clean_word]
                
                # Add only 1-2 quality adjectives (not 4+)
                num_adjectives = min(2, len(adjectives), max_adjectives - adjective_count)
                selected_adjectives = random.sample(adjectives, num_adjectives)
                
                # Ensure natural order: [adjective(s)] [noun]
                enhanced_words.extend(selected_adjectives)
                adjective_count += num_adjectives
            
            enhanced_words.append(word)
            i += 1
        
        # Post-process for grammar quality
        raw_output = " ".join(enhanced_words)
        return self.apply_grammar_quality(raw_output)
    
    def apply_grammar_quality(self, text):
        """Apply grammar quality fixes"""
        # Fix sentence structure
        sentences = re.split(r'[.!?]+', text)
        quality_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # Ensure proper sentence structure
                words = sentence.split()
                if len(words) > 0:
                    # Capitalize first word
                    words[0] = words[0].capitalize()
                    
                    # Remove excessive adjective clustering
                    if len(words) > 8:  # If sentence is too long
                        # Keep core meaning, remove redundancy
                        core_words = self.extract_core_meaning(words)
                        sentence = " ".join(core_words)
                    else:
                        sentence = " ".join(words)
                
                quality_sentences.append(sentence)
        
        # Join with proper punctuation
        quality_text = ". ".join(quality_sentences)
        if quality_text and not quality_text.endswith('.'):
            quality_text += '.'
            
        return quality_text
    
    def extract_core_meaning(self, words):
        """Extract core meaning from wordy sentences"""
        nouns = ['car', 'person', 'building', 'tree', 'city', 'room', 'lights']
        verbs = ['driving', 'dancing', 'walking', 'flying', 'shining']
        
        # Keep nouns, verbs, and 1-2 adjectives per noun
        core_words = []
        adjective_buffer = []
        
        for word in words:
            clean_word = word.lower().strip('.,')
            
            if clean_word in nouns:
                # Add buffered adjectives (max 2) then the noun
                core_words.extend(adjective_buffer[:2])
                core_words.append(word)
                adjective_buffer = []
            elif clean_word in verbs:
                core_words.append(word)
            else:
                # Assume it's an adjective, buffer it
                adjective_buffer.append(word)
        
        return core_words

# Initialize quality-fixed system
quality_system = QualityFixedVisualSystem()

@app.post("/describe/scene")
async def describe_scene(request: SceneDescriptionRequest):
    """Generate quality-enhanced descriptions"""
    try:
        start_time = time.time()
        
        if request.enhance_adjectives and request.grammar_constraint:
            enhanced_desc = quality_system.enhance_with_quality_constraints(
                request.scene_description, 
                request.adjective_density
            )
        else:
            enhanced_desc = request.scene_description
        
        processing_time = time.time() - start_time
        
        return {
            "enhanced_description": enhanced_desc,
            "original_description": request.scene_description,
            "quality_notes": "Grammar constraints applied to prevent keyword stuffing",
            "processing_time": processing_time,
            "version": "3.1.0-quality-fixed"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Description generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 STARTING QUALITY-FIXED VISUAL NARRATOR API...")
    print("📍 API: http://localhost:8003")
    print("🎯 FOCUS: Natural language quality over adjective density")
    uvicorn.run(app, host="0.0.0.0", port=8003)
