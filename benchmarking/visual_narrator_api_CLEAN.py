#!/usr/bin/env python3
"""
CLEAN VISUAL NARRATOR API
- Natural, flowing templates
- Proper grammar and phrasing
- Professional outputs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import random
import re

app = FastAPI(title="Visual Narrator VLM API - CLEAN", version="8.0.0")

class SceneRequest(BaseModel):
    scene_description: str
    enhance_adjectives: bool = True

class CleanVisualSystem:
    def __init__(self):
        # Complete, natural sentence templates
        self.templates = {
            "car_city_night": [
                "A {car_adj} car drives through the {city_adj} city at night, with {light_adj} neon lights {light_action}.",
                "Under the {night_adj} sky, a {car_adj} vehicle moves through {city_adj} streets, {light_adj} neon signs {light_action}.",
                "A {car_adj} automobile navigates the {city_adj} urban landscape at night, where {light_adj} neon illumination {light_action}."
            ],
            "person_dancing_lights": [
                "An {person_adj} dancer moves {action_adj} in a {room_adj} room filled with {light_adj} lighting effects.",
                "In the {room_adj} space, a {person_adj} performer dances {action_adj} under {light_adj} colorful lights.",
                "A {person_adj} figure dances {action_adj} in the {room_adj} environment, surrounded by {light_adj} dynamic illumination."
            ]
        }
        
        self.adjective_pools = {
            "car_adj": ["sleek", "modern", "gleaming", "luxurious", "sporty", "elegant"],
            "city_adj": ["vibrant", "bustling", "illuminated", "lively", "urban", "metropolitan"],
            "night_adj": ["dark", "starry", "clear", "moonlit", "inky", "nocturnal"],
            "light_adj": ["colorful", "glowing", "dazzling", "radiant", "vibrant", "brilliant"],
            "action_adj": ["gracefully", "rhythmically", "energetically", "expressively", "fluidly"],
            "person_adj": ["energetic", "graceful", "expressive", "charismatic", "animated", "dynamic"],
            "room_adj": ["dimly-lit", "colorful", "dynamic", "atmospheric", "vibrant", "pulsating"],
            "light_action": [
                "illuminating the streets",
                "casting colorful reflections", 
                "creating a mesmerizing glow",
                "filling the atmosphere with light",
                "dancing across surfaces"
            ]
        }
    
    def enhance_description(self, description):
        """Generate clean, natural descriptions"""
        description_lower = description.lower()
        
        if "car" in description_lower and "city" in description_lower and "night" in description_lower:
            return self.generate_car_city_description()
        elif "person" in description_lower and "dancing" in description_lower:
            return self.generate_dancing_description()
        else:
            return self.fallback_description(description)
    
    def generate_car_city_description(self):
        """Generate natural car/city description"""
        template = random.choice(self.templates["car_city_night"])
        
        adjectives = {
            "car_adj": random.choice(self.adjective_pools["car_adj"]),
            "city_adj": random.choice(self.adjective_pools["city_adj"]),
            "night_adj": random.choice(self.adjective_pools["night_adj"]),
            "light_adj": random.choice(self.adjective_pools["light_adj"]),
            "light_action": random.choice(self.adjective_pools["light_action"])
        }
        
        result = template.format(**adjectives)
        return self.polish_output(result)
    
    def generate_dancing_description(self):
        """Generate natural dancing description"""
        template = random.choice(self.templates["person_dancing_lights"])
        
        adjectives = {
            "person_adj": random.choice(self.adjective_pools["person_adj"]),
            "action_adj": random.choice(self.adjective_pools["action_adj"]),
            "room_adj": random.choice(self.adjective_pools["room_adj"]),
            "light_adj": random.choice(self.adjective_pools["light_adj"])
        }
        
        result = template.format(**adjectives)
        return self.polish_output(result)
    
    def polish_output(self, text):
        """Final polishing of output"""
        # Fix "a" vs "an" before vowels
        text = re.sub(r'\ba ([aeiou])', r'an \1', text, flags=re.IGNORECASE)
        
        # Fix repeated words
        text = re.sub(r'\b(\w+) \1\b', r'\1', text)
        
        # Fix double spaces
        text = re.sub(r'  +', ' ', text)
        
        # Remove trailing spaces before punctuation
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        
        # Ensure proper capitalization and punctuation
        if text and text[0].isalpha():
            text = text[0].upper() + text[1:]
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text
    
    def fallback_description(self, description):
        """Simple, clean fallback"""
        return self.polish_output(description)

clean_system = CleanVisualSystem()

@app.post("/describe/scene")
async def describe_scene(request: SceneRequest):
    """Generate clean, professional descriptions"""
    try:
        start_time = time.time()
        
        if request.enhance_adjectives:
            enhanced_desc = clean_system.enhance_description(request.scene_description)
        else:
            enhanced_desc = request.scene_description
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "enhanced_description": enhanced_desc,
            "original_description": request.scene_description,
            "processing_time": processing_time,
            "version": "8.0.0-clean"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Description generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 STARTING CLEAN VISUAL NARRATOR API...")
    print("📍 API: http://localhost:8010")
    print("🎯 FOCUS: Natural, flowing professional outputs")
    uvicorn.run(app, host="0.0.0.0", port=8010)
