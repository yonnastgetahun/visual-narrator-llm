emotional_scenes = [
    # Action scenes
    {"description": "car chase with explosions", "type": "action", "intensity": 0.9},
    {"description": "martial arts fight in rain", "type": "action", "intensity": 0.8},
    {"description": "helicopter escape from building", "type": "action", "intensity": 0.95},
    
    # Drama scenes  
    {"description": "romantic sunset on beach", "type": "drama", "intensity": 0.7},
    {"description": "emotional hospital confession", "type": "drama", "intensity": 0.85},
    {"description": "tense courtroom verdict", "type": "drama", "intensity": 0.75},
    
    # Comedy scenes
    {"description": "comedic slip on banana", "type": "comedy", "intensity": 0.6},
    {"description": "awkward first date mishap", "type": "comedy", "intensity": 0.5},
    {"description": "office prank backfires", "type": "comedy", "intensity": 0.4},
    
    # Horror scenes
    {"description": "dark haunted house exploration", "type": "horror", "intensity": 0.8},
    {"description": "jump scare in mirror", "type": "horror", "intensity": 0.9},
    
    # Documentary scenes
    {"description": "nature documentary wildlife", "type": "documentary", "intensity": 0.4},
    {"description": "historical reenactment", "type": "documentary", "intensity": 0.3}
]

print(f"Expanded emotional dataset: {len(emotional_scenes)} scenes")
print("\nScene types distribution:")
from collections import Counter
scene_types = Counter([scene['type'] for scene in emotional_scenes])
for scene_type, count in scene_types.items():
    print(f"- {scene_type}: {count} scenes")

# Save to JSON for future training
import json
with open('emotional_scenes_dataset.json', 'w') as f:
    json.dump(emotional_scenes, f, indent=2)
print("\n✅ Dataset saved to emotional_scenes_dataset.json")
