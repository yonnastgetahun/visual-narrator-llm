import json
import random
from PIL import Image
import os

def create_visual_grounding_dataset():
    """Create dataset for visual grounding training"""
    
    print("🔄 CREATING VISUAL GROUNDING DATASET")
    print("=" * 50)
    
    # Enhanced adjective pool with object-specific mappings
    object_adjective_mapping = {
        "person": ["elegant", "expressive", "dynamic", "poised", "animated"],
        "car": ["sleek", "streamlined", "powerful", "gleaming", "modern"],
        "building": ["majestic", "imposing", "architectural", "towering", "historic"],
        "tree": ["lush", "verdant", "towering", "ancient", "leafy"],
        "sky": ["dramatic", "expansive", "atmospheric", "luminous", "cloudy"],
        "water": ["glistening", "tranquil", "rippling", "crystal", "flowing"],
        "animal": ["graceful", "majestic", "wild", "poised", "curious"]
    }
    
    spatial_relations = [
        "in front of", "behind", "next to", "above", "below", 
        "between", "surrounded by", "near", "far from", "across from"
    ]
    
    # Create enhanced training examples
    enhanced_dataset = []
    
    # Example templates for different improvement areas
    templates = [
        # Object-specific descriptions
        "a {adjective1} {object1} in a {adjective2} {scene_type}",
        "a {adjective1} {object1} {spatial_relation} a {adjective2} {object2}",
        "multiple {adjective1} {object1}s in a {adjective2} {scene_type}",
        
        # Spatial relationships
        "a {adjective1} {object1} positioned {spatial_relation} the {adjective2} {object2}",
        "the {adjective1} {object1} stands {spatial_relation} the {adjective2} {object2}",
        
        # Complex scenes
        "a {scene_type} featuring a {adjective1} {object1} and a {adjective2} {object2}",
        "multiple objects including a {adjective1} {object1} and {adjective2} {object2}"
    ]
    
    objects = list(object_adjective_mapping.keys())
    scene_types = ["urban landscape", "natural setting", "indoor environment", "street scene"]
    
    # Generate diverse examples
    for i in range(1000):  # Start with 1000 enhanced examples
        template = random.choice(templates)
        
        # Fill template
        if "{object1}" in template and "{object2}" in template:
            obj1 = random.choice(objects)
            obj2 = random.choice([o for o in objects if o != obj1])
            adj1 = random.choice(object_adjective_mapping[obj1])
            adj2 = random.choice(object_adjective_mapping[obj2])
            spatial = random.choice(spatial_relations)
            
            caption = template.format(
                adjective1=adj1, adjective2=adj2,
                object1=obj1, object2=obj2,
                spatial_relation=spatial,
                scene_type=random.choice(scene_types)
            )
        else:
            # Simple object description
            obj = random.choice(objects)
            adj = random.choice(object_adjective_mapping[obj])
            caption = template.format(
                adjective1=adj, adjective2=random.choice(["vivid", "dramatic", "serene"]),
                object1=obj,
                scene_type=random.choice(scene_types)
            )
        
        enhanced_dataset.append({
            "caption": caption,
            "objects": [obj1] if 'obj1' in locals() else [random.choice(objects)],
            "adjectives": [adj1, adj2] if 'adj1' in locals() and 'adj2' in locals() else [adj],
            "has_spatial_relations": "spatial_relation" in template,
            "training_focus": "visual_grounding"
        })
    
    # Save enhanced dataset
    output_path = "phase8/visual_grounding_dataset.json"
    os.makedirs("phase8", exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(enhanced_dataset, f, indent=2)
    
    print(f"✅ Enhanced dataset created: {output_path}")
    print(f"📊 Dataset Statistics:")
    print(f"   - Total examples: {len(enhanced_dataset)}")
    print(f"   - Object types: {len(objects)}")
    print(f"   - Spatial relations: {len(spatial_relations)}")
    print(f"   - Average adjectives per caption: {sum(len(item['adjectives']) for item in enhanced_dataset) / len(enhanced_dataset):.2f}")
    
    # Show examples
    print(f"📝 Example Enhanced Captions:")
    for i in range(5):
        print(f"   {i+1}. {enhanced_dataset[i]['caption']}")
    
    return enhanced_dataset

if __name__ == "__main__":
    create_visual_grounding_dataset()
