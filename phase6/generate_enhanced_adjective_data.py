import json

def create_adjective_domination_data():
    """Create comprehensive adjective training data"""
    
    adjective_categories = {
        "visual_quality": ["beautiful", "stunning", "vibrant", "dramatic", "serene", "majestic", "picturesque", "breathtaking", "spectacular"],
        "size_scale": ["massive", "towering", "spacious", "compact", "vast", "tiny", "enormous", "gigantic", "colossal", "immense"],
        "color_palette": ["colorful", "vibrant", "bright", "muted", "pastel", "earthy", "rich", "vivid", "saturated", "monochromatic"],
        "texture_material": ["sleek", "textured", "glossy", "matte", "rough", "smooth", "porous", "gritty", "polished", "coarse"],
        "atmosphere_mood": ["peaceful", "chaotic", "romantic", "mysterious", "joyful", "melancholy", "energetic", "tranquil", "lively", "somber"],
        "architectural": ["ornate", "minimalist", "historic", "modern", "futuristic", "traditional", "contemporary", "elegant", "rustic"],
        "natural": ["lush", "barren", "pristine", "wild", "cultivated", "dense", "sparse", "fertile", "arid", "verdant"]
    }
    
    # URBAN FOCUS - Specific to our performance gap
    urban_adjectives = {
        "buildings": ["skyscraper", "high-rise", "brownstone", "modern", "historic", "commercial", "residential"],
        "streets": ["bustling", "crowded", "empty", "winding", "wide", "narrow", "pedestrian"],
        "atmosphere": ["vibrant", "gritty", "sophisticated", "touristy", "residential", "commercial", "industrial"]
    }
    
    training_examples = []
    
    # ===== TIER 1: BASIC ADJECTIVE FOUNDATION =====
    print("🏗️  Creating Tier 1: Basic Adjective Foundation...")
    
    base_scenes = [
        "a sunset over mountains", "a city street with buildings", "a forest with trees",
        "a beach with ocean", "a kitchen with appliances", "an urban park with people",
        "a sports stadium with athletes", "a classroom with students", "a concert venue with musicians"
    ]
    
    for scene in base_scenes:
        for category, adjectives in adjective_categories.items():
            for adjective in adjectives[:3]:  # Use top 3 from each category
                templates = [
                    f"Describe this image: {adjective} {scene}",
                    f"This picture shows {adjective} {scene}",
                    f"Generate a caption for this {adjective} scene: {scene}",
                    f"What do you see in this {adjective} image: {scene}",
                    f"Describe the {adjective} qualities of this scene: {scene}"
                ]
                training_examples.extend(templates)
    
    # ===== TIER 2: MULTI-ADJECTIVE STACKING =====
    print("🏗️  Creating Tier 2: Multi-Adjective Stacking...")
    
    multi_adjective_scenes = [
        ("a beautiful vibrant sunset over majestic mountains", "nature"),
        ("a busy chaotic city street with tall modern buildings", "urban"),
        ("a peaceful serene forest with lush green trees", "nature"), 
        ("a colorful energetic sports event with enthusiastic athletes", "people"),
        ("a spacious modern kitchen with sleek stainless appliances", "indoor"),
        ("a historic ornate building with intricate architectural details", "urban"),
        ("a vast barren desert landscape with dramatic rock formations", "nature"),
        ("a crowded bustling market with colorful vibrant stalls", "urban"),
        ("a quiet empty classroom with neat organized desks", "indoor"),
        ("a dramatic stormy ocean with powerful crashing waves", "nature")
    ]
    
    for scene_desc, category in multi_adjective_scenes:
        templates = [
            f"Describe this image: {scene_desc}",
            f"This photograph captures {scene_desc}",
            f"Generate a detailed description: {scene_desc}",
            f"What makes this scene visually interesting: {scene_desc}",
            f"Create an audio description for: {scene_desc}"
        ]
        training_examples.extend(templates)
    
    # ===== TIER 3: URBAN SCENE FOCUS (Addressing our 75% gap) =====
    print("🏗️  Creating Tier 3: Urban Scene Focus...")
    
    urban_scenes = [
        "downtown skyscraper district at night",
        "busy commercial street with shops and cafes", 
        "quiet residential neighborhood with houses",
        "urban park with trees and walking paths",
        "industrial area with warehouses and factories",
        "public square with fountains and monuments",
        "suburban shopping center with parking lots",
        "historic city center with old buildings",
        "modern business district with glass towers",
        "public transportation hub with buses and trains"
    ]
    
    for scene in urban_scenes:
        # Add multiple urban-focused adjective combinations
        urban_templates = [
            f"Describe this urban scene: {scene}",
            f"Generate a detailed description of this city view: {scene}",
            f"Create an audio description for this metropolitan area: {scene}",
            f"Describe the architecture and atmosphere of: {scene}",
            f"What visual elements stand out in this cityscape: {scene}"
        ]
        training_examples.extend(urban_templates)
    
    # ===== TIER 4: COMPLEX SCENE DECOMPOSITION =====
    print("🏗️  Creating Tier 4: Complex Scene Decomposition...")
    
    complex_scenes = [
        ("A group of diverse people interacting in a public space with multiple activities happening simultaneously", "complex_people"),
        ("An intricate architectural interior with multiple rooms, furniture, and decorative elements", "complex_interior"),
        ("A layered landscape with foreground, midground, and background elements creating depth", "complex_landscape"),
        ("A detailed street scene with vehicles, pedestrians, buildings, and urban infrastructure", "complex_urban"),
        ("An event scene with performers, audience, lighting, and stage elements", "complex_event")
    ]
    
    for scene_desc, scene_type in complex_scenes:
        complex_templates = [
            f"Break down this complex scene into descriptive elements: {scene_desc}",
            f"Describe the multiple components of this detailed image: {scene_desc}", 
            f"Create a comprehensive audio description for this intricate scene: {scene_desc}",
            f"Capture all the visual details in this complex scenario: {scene_desc}"
        ]
        training_examples.extend(complex_templates)
    
    print(f"✅ Created {len(training_examples)} comprehensive adjective-focused training examples")
    return training_examples

def save_datasets(training_examples):
    """Save enhanced datasets in separate files"""
    
    # Split examples by type (simple logic based on content)
    basic = [ex for ex in training_examples if "adjective" not in ex.lower() or ex.count(' ') < 8]
    multi_adj = [ex for ex in training_examples if ex.count(' ') >= 8 and ex.count(' ') < 12]
    complex_scenes = [ex for ex in training_examples if ex.count(' ') >= 12]
    
    # Save separate datasets
    with open('adjective_data/forced_adjectives.jsonl', 'w') as f:
        for example in basic:
            f.write(json.dumps({"text": example}) + '\n')
    
    with open('adjective_data/multi_adjective_stacking.jsonl', 'w') as f:
        for example in multi_adj:
            f.write(json.dumps({"text": example}) + '\n')
    
    with open('adjective_data/complex_scenes.jsonl', 'w') as f:
        for example in complex_scenes:
            f.write(json.dumps({"text": example}) + '\n')
    
    print(f"📊 Saved datasets:")
    print(f"   - Basic adjectives: {len(basic)} examples")
    print(f"   - Multi-adjective: {len(multi_adj)} examples") 
    print(f"   - Complex scenes: {len(complex_scenes)} examples")
    print(f"   - Total: {len(training_examples)} examples")

# Generate and save enhanced data
if __name__ == "__main__":
    data = create_adjective_domination_data()
    save_datasets(data)
    
    # Verify
    import subprocess
    print("\n📈 Verification:")
    for dataset in ['forced_adjectives', 'multi_adjective_stacking', 'complex_scenes']:
        result = subprocess.run(['wc', '-l', f'adjective_data/{dataset}.jsonl'], 
                              capture_output=True, text=True)
        print(f"   {dataset}: {result.stdout.strip()}")
