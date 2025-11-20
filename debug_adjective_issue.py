#!/usr/bin/env python3
"""
Debug script to identify adjective density issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.adjective_enhancer import enhance_description
from app.core.spatial_analyzer import analyze_spatial_relations
from app.core.integration_engine import IntegratedVLM

def test_direct_components():
    print("🔍 TESTING CORE COMPONENTS DIRECTLY")
    print("=" * 60)
    
    test_scenes = [
        "a car near a building",
        "a beautiful sunset over majestic mountains", 
        "a person walking a dog in a park"
    ]
    
    # Test adjective enhancer directly
    print("\n🧪 DIRECT ADJECTIVE ENHANCER TEST:")
    for scene in test_scenes:
        enhanced = enhance_description(scene, adjective_density=1.0)
        print(f"Input: {scene}")
        print(f"Enhanced: {enhanced}")
        print(f"Adjective count: {len([word for word in enhanced.split() if word in ['beautiful', 'majestic', 'vibrant', 'serene', 'dramatic', 'elegant']])}")
        print("-" * 40)
    
    # Test integrated model
    print("\n🧪 INTEGRATED MODEL TEST:")
    try:
        model = IntegratedVLM()
        for scene in test_scenes:
            result = model.process_scene(
                scene_description=scene,
                enhance_adjectives=True,
                adjective_density=1.0,
                include_spatial=True
            )
            print(f"Input: {scene}")
            print(f"Output: {result['enhanced_description']}")
            print(f"Metrics: {result['metrics']}")
            print("-" * 40)
    except Exception as e:
        print(f"❌ Integrated model error: {e}")
    
    # Test spatial analyzer
    print("\n🧪 SPATIAL ANALYZER TEST:")
    for scene in test_scenes:
        spatial_result = analyze_spatial_relations(scene)
        print(f"Input: {scene}")
        print(f"Spatial relations: {spatial_result}")
        print("-" * 40)

if __name__ == "__main__":
    test_direct_components()
