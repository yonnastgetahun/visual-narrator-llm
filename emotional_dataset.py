emotional_scenes = [
    {"description": "car chase with explosions", "type": "action", "intensity": 0.9},
    {"description": "romantic sunset on beach", "type": "drama", "intensity": 0.7},
    {"description": "comedic slip on banana", "type": "comedy", "intensity": 0.6},
]

print("Emotional dataset created with", len(emotional_scenes), "scenes")
for scene in emotional_scenes:
    print(f"- {scene['type']}: {scene['description']} (intensity: {scene['intensity']})")
