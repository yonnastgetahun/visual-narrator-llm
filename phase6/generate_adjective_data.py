def create_adjective_domination_data():
    """Create training data that forces adjective usage"""
    
    adjective_categories = {
        "visual_quality": ["beautiful", "stunning", "vibrant", "dramatic", "serene", "majestic", "picturesque"],
        "size_scale": ["massive", "towering", "spacious", "compact", "vast", "tiny", "enormous"],
        "color_palette": ["colorful", "vibrant", "bright", "muted", "pastel", "earthy", "rich"],
        "texture_material": ["sleek", "textured", "glossy", "matte", "rough", "smooth", "porous"],
        "atmosphere_mood": ["peaceful", "chaotic", "romantic", "mysterious", "joyful", "melancholy", "energetic"]
    }
    
    training_examples = []
    
    # Template 1: Forced adjective insertion
    base_scenes = [
        "a sunset over mountains", "a city street with buildings", "a forest with trees",
        "a beach with ocean", "a kitchen with appliances"
    ]
    
    for scene in base_scenes:
        for category, adjectives in adjective_categories.items():
            for adjective in adjectives[:2]:  # Use top 2 from each category
                templates = [
                    f"Describe this image: {adjective} {scene}",
                    f"This picture shows {adjective} {scene}",
                    f"Generate a caption for this {adjective} scene: {scene}"
                ]
                training_examples.extend(templates)
    
    print(f"✅ Created {len(training_examples)} adjective-focused training examples")
    return training_examples

# Test it
if __name__ == "__main__":
    data = create_adjective_domination_data()
    
    # Save to your adjective_data directory
    with open('adjective_data/forced_adjectives.jsonl', 'w') as f:
        for example in data:
            f.write('{"text": "' + example + '"}\\n')
    
    print("✅ Saved to adjective_data/forced_adjectives.jsonl")
