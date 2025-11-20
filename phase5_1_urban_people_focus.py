#!/usr/bin/env python3
"""
PHASE 5.1: URBAN & PEOPLE SCENE FOCUS
Fix the weak areas identified in Phase 4 benchmarking
"""

print("🚀 PHASE 5.1: URBAN & PEOPLE SCENE FOCUS")
print("=" * 50)

def create_specialized_training_data():
    """Create training data focused on urban and people scenes"""
    print("📊 Creating specialized training data for weak areas...")
    
    specialized_descriptions = []
    
    # URBAN SCENES (Current weakness: 0% success)
    urban_scenes = [
        # City streets and transportation
        "a busy city street with cars, buses, and pedestrians crossing at intersections",
        "a downtown area with tall skyscrapers, traffic lights, and people walking on sidewalks",
        "a subway station with trains arriving, commuters waiting, and digital schedule displays",
        "a bus stop with people waiting, buses pulling in, and city buildings in background",
        "a traffic jam on a highway with multiple lanes of cars and trucks stopped",
        "a city park surrounded by buildings with people walking dogs and sitting on benches",
        "a shopping district with storefronts, signs, and crowds of people browsing",
        "a construction site with cranes, workers in hard hats, and heavy machinery operating",
        
        # Urban architecture
        "a modern office building with glass windows, corporate logos, and entrance doors",
        "a historic building with ornate architecture, large windows, and decorative elements",
        "an apartment complex with multiple floors, balconies, and residents entering/exiting",
        "a hotel with a grand entrance, doorman, and luxury vehicles parked outside",
        "a government building with columns, flags, and people conducting official business",
        "a school building with playground, buses, and children arriving for classes",
        "a hospital with emergency entrance, ambulances, and medical staff moving about",
        "a restaurant with outdoor seating, customers dining, and waitstaff serving food",
        
        # Urban activities
        "a street market with vendors selling goods, customers browsing, and colorful displays",
        "a public square with fountains, benches, and people gathering for events",
        "a parking garage with multiple levels, cars parking, and payment machines",
        "a bridge over a river with traffic flowing and city skyline in background",
        "a train station with platforms, arriving trains, and passengers with luggage"
    ]
    
    # PEOPLE ACTIVITIES (Current: 50% success - needs improvement)
    people_scenes = [
        # Sports and recreation
        "a basketball game with players dribbling, shooting, and referees officiating",
        "a soccer match with athletes running, kicking balls, and coaches directing",
        "a swimming pool with people swimming laps, diving, and lifeguards watching",
        "a gym with people exercising on machines, lifting weights, and trainers assisting",
        "a yoga class with participants stretching, instructor demonstrating, and mats",
        "a marathon with runners competing, spectators cheering, and finish line banner",
        "a tennis match with players serving, volleying, and ball kids retrieving",
        "a baseball game with pitcher throwing, batter swinging, and fielders ready",
        
        # Education and work
        "a classroom with students listening, teacher writing on board, and desks",
        "a university lecture with professor presenting, students taking notes, projector",
        "an office meeting with colleagues discussing, presentation screen, notebooks",
        "a laboratory with scientists conducting experiments, equipment, safety goggles",
        "a library with people reading, studying at tables, and bookshelves with books",
        "a workshop with craftspeople building, tools organized, and projects in progress",
        "a conference with attendees networking, speakers presenting, exhibition booths",
        "a coffee shop with people working on laptops, baristas preparing drinks",
        
        # Social and family
        "a family dinner with relatives eating, talking, and passing food around table",
        "a birthday party with guests celebrating, cake with candles, presents opened",
        "a wedding ceremony with bride and groom exchanging vows, guests watching",
        "a group of friends hiking on trail, backpacks, scenic views in background",
        "a team collaborating in conference room, whiteboard with diagrams, computers",
        "a community event with volunteers organizing, participants engaging, banners",
        "a musical performance with musicians playing instruments, audience applauding",
        "a art class with students painting, easels, instructor giving guidance"
    ]
    
    # Templates for natural language variation
    templates = [
        "This image shows {scene}",
        "In this picture, we see {scene}",
        "The image depicts {scene}",
        "Here is a photo of {scene}",
        "This photograph captures {scene}",
        "We can see {scene} in this image",
        "The scene shows {scene}",
        "Displayed here is {scene}",
        "Captured in this image is {scene}",
        "This visual contains {scene}"
    ]
    
    # Generate training data with focus on weak areas
    print("🏙️  Generating urban training data...")
    for scene in urban_scenes:
        for template in templates:
            specialized_descriptions.append(template.format(scene=scene))
        # Extra emphasis on urban scenes
        specialized_descriptions.append(f"Describe this urban scene: {scene}")
        specialized_descriptions.append(f"What do you see in this city image: {scene}")
    
    print("👥 Generating people activity training data...")
    for scene in people_scenes:
        for template in templates:
            specialized_descriptions.append(template.format(scene=scene))
        # Extra emphasis on people activities
        specialized_descriptions.append(f"Describe this people activity: {scene}")
        specialized_descriptions.append(f"What is happening in this image: {scene}")
    
    # Keep some nature scenes for balance (our strength)
    nature_scenes = [
        "a beautiful sunset over mountains with clouds reflecting warm colors",
        "a peaceful forest with sunlight filtering through trees creating patterns",
        "a tropical beach with white sand, palm trees, and turquoise ocean waves"
    ]
    
    print("🌲 Adding balanced nature scenes...")
    for scene in nature_scenes:
        for template in templates[:5]:  # Fewer templates for nature (our strength)
            specialized_descriptions.append(template.format(scene=scene))
    
    print(f"✅ Created {len(specialized_descriptions)} specialized training examples")
    print(f"   - Urban scenes: {len(urban_scenes) * len(templates) + len(urban_scenes) * 2}")
    print(f"   - People activities: {len(people_scenes) * len(templates) + len(people_scenes) * 2}")
    print(f"   - Nature scenes: {len(nature_scenes) * 5}")
    
    return specialized_descriptions

def create_phase5_1_plan():
    """Create the Phase 5.1 implementation plan"""
    print("\n📋 PHASE 5.1 IMPLEMENTATION PLAN")
    print("=" * 40)
    
    plan = {
        "week_1_goals": [
            "Collect 500+ urban scene descriptions",
            "Gather 500+ people activity descriptions", 
            "Create specialized training mixture",
            "Train focused model on weak areas",
            "Validate urban scene performance > 50%",
            "Validate people scene performance > 70%"
        ],
        "technical_approach": [
            "Use proven OPT-2.7B + LoRA pipeline",
            "Focus training on urban/people data",
            "Maintain nature scene strength",
            "Rapid iteration with validation",
            "Early stopping to prevent overfitting"
        ],
        "success_metrics": [
            "Urban scenes: > 50% success (from 0%)",
            "People scenes: > 70% success (from 50%)", 
            "Overall success: > 65% (from 50%)",
            "No regression in nature scenes (maintain 100%)"
        ]
    }
    
    for section, items in plan.items():
        print(f"\n{section.replace('_', ' ').title()}:")
        for item in items:
            print(f"  • {item}")
    
    return plan

def main():
    """Main Phase 5.1 planning"""
    try:
        # Create specialized training data
        training_data = create_specialized_training_data()
        
        # Create implementation plan
        plan = create_phase5_1_plan()
        
        # Save training data for immediate use
        import os
        os.makedirs("./phase5", exist_ok=True)
        
        with open("./phase5/specialized_training_data.json", "w") as f:
            import json
            json.dump({
                "training_examples": training_data,
                "total_count": len(training_data),
                "focus_areas": ["urban_scenes", "people_activities"],
                "phase": "5.1",
                "timestamp": str(__import__('datetime').datetime.now())
            }, f, indent=2)
        
        print(f"\n💾 Specialized training data saved: {len(training_data)} examples")
        print("🚀 PHASE 5.1 READY FOR EXECUTION!")
        print("\n🎯 Next: Run focused training on urban and people scenes")
        print("   Command: python3 phase5_1_focused_training.py")
        
    except Exception as e:
        print(f"❌ Phase 5.1 planning error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
