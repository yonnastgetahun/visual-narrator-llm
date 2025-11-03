import json

def create_adjective_domination_data():
    """Create comprehensive adjective training data with proper categorization"""
    
    adjective_categories = {
        "visual_quality": ["beautiful", "stunning", "vibrant", "dramatic", "serene", "majestic", "picturesque", "breathtaking", "spectacular"],
        "size_scale": ["massive", "towering", "spacious", "compact", "vast", "tiny", "enormous", "gigantic", "colossal", "immense"],
        "color_palette": ["colorful", "vibrant", "bright", "muted", "pastel", "earthy", "rich", "vivid", "saturated", "monochromatic"],
        "texture_material": ["sleek", "textured", "glossy", "matte", "rough", "smooth", "porous", "gritty", "polished", "coarse"],
        "atmosphere_mood": ["peaceful", "chaotic", "romantic", "mysterious", "joyful", "melancholy", "energetic", "tranquil", "lively", "somber"],
        "architectural": ["ornate", "minimalist", "historic", "modern", "futuristic", "traditional", "contemporary", "elegant", "rustic"],
        "natural": ["lush", "barren", "pristine", "wild", "cultivated", "dense", "sparse", "fertile", "arid", "verdant"]
    }
    
    basic_examples = []
    multi_adj_examples = []
    complex_examples = []
    
    # ===== TIER 1: BASIC ADJECTIVE FOUNDATION =====
    print("🏗️  Creating Tier 1: Basic Adjective Foundation...")
    
    base_scenes = [
        "a sunset over mountains", "a city street with buildings", "a forest with trees",
        "a beach with ocean", "a kitchen with appliances", "an urban park with people",
        "a sports stadium with athletes", "a classroom with students", "a concert venue with musicians"
    ]
    
    for scene in base_scenes:
        for category, adjectives in adjective_categories.items():
            for adjective in adjectives[:3]:
                templates = [
                    f"Describe this image: {adjective} {scene}",
                    f"This picture shows {adjective} {scene}",
                    f"Generate a caption for this {adjective} scene: {scene}",
                    f"What do you see in this {adjective} image: {scene}",
                    f"Describe the {adjective} qualities of this scene: {scene}"
                ]
                basic_examples.extend(templates)
    
    # ===== TIER 2: MULTI-ADJECTIVE STACKING =====
    print("🏗️  Creating Tier 2: Multi-Adjective Stacking...")
    
    multi_adjective_scenes = [
        "a beautiful vibrant sunset over majestic mountains",
        "a busy chaotic city street with tall modern buildings",
        "a peaceful serene forest with lush green trees", 
        "a colorful energetic sports event with enthusiastic athletes",
        "a spacious modern kitchen with sleek stainless appliances",
        "a historic ornate building with intricate architectural details",
        "a vast barren desert landscape with dramatic rock formations",
        "a crowded bustling market with colorful vibrant stalls",
        "a quiet empty classroom with neat organized desks",
        "a dramatic stormy ocean with powerful crashing waves"
    ]
    
    for scene_desc in multi_adjective_scenes:
        templates = [
            f"Describe this image: {scene_desc}",
            f"This photograph captures {scene_desc}",
            f"Generate a detailed description: {scene_desc}",
            f"What makes this scene visually interesting: {scene_desc}",
            f"Create an audio description for: {scene_desc}"
        ]
        multi_adj_examples.extend(templates)
    
    # ===== TIER 3: COMPLEX SCENE DECOMPOSITION =====
    print("🏗️  Creating Tier 3: Complex Scene Decomposition...")
    
    complex_scenes = [
        "A group of diverse people interacting in a public space with multiple activities happening simultaneously",
        "An intricate architectural interior with multiple rooms, furniture, and decorative elements",
        "A layered landscape with foreground, midground, and background elements creating depth",
        "A detailed street scene with vehicles, pedestrians, buildings, and urban infrastructure",
        "An event scene with performers, audience, lighting, and stage elements"
    ]
    
    for scene_desc in complex_scenes:
        complex_templates = [
            f"Break down this complex scene into descriptive elements: {scene_desc}",
            f"Describe the multiple components of this detailed image: {scene_desc}", 
            f"Create a comprehensive audio description for this intricate scene: {scene_desc}",
            f"Capture all the visual details in this complex scenario: {scene_desc}"
        ]
        complex_examples.extend(complex_templates)
    
    return basic_examples, multi_adj_examples, complex_examples

def save_datasets(basic, multi_adj, complex_scenes):
    """Save properly categorized datasets"""
    
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
    
    print(f"📊 PROPERLY CATEGORIZED DATASETS:")
    print(f"   - Basic adjectives: {len(basic)} examples")
    print(f"   - Multi-adjective: {len(multi_adj)} examples") 
    print(f"   - Complex scenes: {len(complex_scenes)} examples")
    print(f"   - Total: {len(basic) + len(multi_adj) + len(complex_scenes)} examples")

# Generate and save enhanced data
if __name__ == "__main__":
    basic, multi_adj, complex_scenes = create_adjective_domination_data()
    save_datasets(basic, multi_adj, complex_scenes)
    
    # Verify with samples
    print("\n🎯 SAMPLE VERIFICATION:")
    print("BASIC examples:")
    for i in range(2):
        print(f"  {basic[i]}")
    
    print("\nMULTI-ADJECTIVE examples:")
    for i in range(2):
        print(f"  {multi_adj[i]}")
        
    print("\nCOMPLEX examples:")
    for i in range(2):
        print(f"  {complex_scenes[i]}")
