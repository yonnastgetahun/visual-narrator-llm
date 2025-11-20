import json

def test_spatial_dataset():
    """Test if our spatial dataset has the right patterns"""
    
    data = json.load(open("phase8/spatial_intensive_dataset.json"))
    
    spatial_terms = ['front of', 'behind', 'above', 'below', 'between', 'next to', 
                    'adjacent to', 'flanking', 'overlooking', 'underneath', 'facing',
                    'to the right of', 'beneath', 'alongside', 'backing onto', 'opposite']
    
    spatial_count = 0
    adjective_count = 0
    
    for i, item in enumerate(data[:100]):  # Check first 100 examples
        caption = item["caption"].lower()
        
        # Count spatial terms
        has_spatial = any(term in caption for term in spatial_terms)
        if has_spatial: spatial_count += 1
        
        # Count adjectives
        adjectives = ['vivid', 'gleaming', 'rugged', 'tranquil', 'velvety', 'golden', 
                     'majestic', 'luminous', 'expressive', 'sleek', 'towering', 'ancient', 'graceful']
        adj_in_caption = sum(1 for adj in adjectives if adj in caption)
        adjective_count += adj_in_caption
        
        if i < 5:  # Show first 5 examples
            print(f"Example {i+1}: {item['caption']}")
            print(f"  - Has spatial: {has_spatial}")
            print(f"  - Adjectives: {adj_in_caption}")
            print()
    
    print("📊 DATASET ANALYSIS:")
    print(f"   - Spatial examples: {spatial_count}/100 ({spatial_count}%)")
    print(f"   - Avg adjectives per caption: {adjective_count/100:.2f}")
    print(f"   - Dataset quality: {'✅ EXCELLENT' if spatial_count >= 95 else '⚠️ NEEDS IMPROVEMENT'}")

if __name__ == "__main__":
    test_spatial_dataset()
