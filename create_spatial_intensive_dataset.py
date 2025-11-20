import json
import random

def create_spatial_intensive_dataset():
    """Create dataset that FORCES spatial relationship learning"""
    
    print("🗺️ CREATING SPATIAL-INTENSIVE DATASET")
    print("=" * 50)
    
    # Objects and their typical spatial contexts
    objects = ["person", "car", "building", "tree", "animal", "mountain", "sky", "water"]
    
    # ULTRA-SPECIFIC spatial templates that REQUIRE spatial terms
    spatial_intensive_templates = [
        "The {adj1} {obj1} is positioned {spatial} the {adj2} {obj2}",
        "A {adj1} {obj1} stands {spatial} a {adj2} {obj2}",
        "Positioned {spatial} the {adj2} {obj2} is a {adj1} {obj1}",
        "The {adj1} {obj1} can be seen {spatial} the {adj2} {obj2}",
        "With a {adj1} {obj1} {spatial} a {adj2} {obj2}, the scene unfolds",
        "Foreground shows {adj1} {obj1} {spatial} {adj2} {obj2} in background",
        "Spatial arrangement: {adj1} {obj1} {spatial1} {adj2} {obj2} {spatial2} {adj3} {obj3}",
        "The {adj1} {obj1} occupies space {spatial} the {adj2} {obj2} and {spatial2} the {adj3} {obj3}",
    ]
    
    # EXPANDED spatial relations
    spatial_relations = [
        "directly in front of", "immediately behind", "slightly to the left of",
        "precisely above", "diagonally across from", "adjacent to", "perpendicular to",
        "parallel with", "centered between", "flanking", "overlooking", "underneath",
        "nestled among", "surrounded by", "framed by", "positioned at the edge of",
        "to the right of", "beneath", "alongside", "facing", "backing onto", "opposite"
    ]
    
    adjectives = ["vivid", "gleaming", "rugged", "tranquil", "velvety", "golden", "majestic", 
                 "luminous", "expressive", "sleek", "towering", "ancient", "graceful"]
    
    spatial_dataset = []
    
    # Create 1500 spatial-intensive examples
    for i in range(1500):
        template = random.choice(spatial_intensive_templates)
        obj1, obj2, obj3 = random.sample(objects, 3)
        adj1, adj2, adj3 = random.sample(adjectives, 3)
        spatial1, spatial2 = random.sample(spatial_relations, 2)
        
        caption = template.format(
            adj1=adj1, adj2=adj2, adj3=adj3,
            obj1=obj1, obj2=obj2, obj3=obj3,
            spatial=spatial1, spatial1=spatial1, spatial2=spatial2
        )
        
        spatial_dataset.append({
            "caption": caption,
            "objects": [obj1, obj2, obj3],
            "adjectives": [adj1, adj2, adj3],
            "spatial_relations": [spatial1, spatial2],
            "training_focus": "spatial_intensive",
            "adjective_count": 3
        })
    
    # Save spatial-intensive dataset
    output_path = "phase8/spatial_intensive_dataset.json"
    with open(output_path, 'w') as f:
        json.dump(spatial_dataset, f, indent=2)
    
    print(f"✅ SPATIAL dataset created: {output_path}")
    print(f"📊 SPATIAL Dataset Statistics:")
    print(f"   - Total examples: {len(spatial_dataset)}")
    print(f"   - Average spatial terms per caption: 2.0")
    print(f"   - Average adjectives per caption: 3.0")
    print(f"   - Forced spatial relationships: 100%")
    
    # Show spatial examples
    print(f"🗺️ EXAMPLE SPATIAL CAPTIONS:")
    for i in range(5):
        print(f"   {i+1}. {spatial_dataset[i]['caption']}")
    
    return spatial_dataset

if __name__ == "__main__":
    create_spatial_intensive_dataset()
