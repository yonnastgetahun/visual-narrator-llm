#!/usr/bin/env python3
"""
Setup Hugging Face Repository for Visual Narrator LLM
"""

import os
import json
import shutil
from pathlib import Path

def create_huggingface_repo():
    """Create complete Hugging Face repository structure"""
    print("🚀 Setting up Hugging Face repository...")
    
    repo_dir = Path("visual-narrator-opt-2.7b")
    repo_dir.mkdir(exist_ok=True)
    
    # Copy model files
    model_src = Path("./phase5/focused_model")
    if model_src.exists():
        print("📁 Copying model files...")
        for item in model_src.iterdir():
            if item.is_file():
                shutil.copy2(item, repo_dir / item.name)
            else:
                shutil.copytree(item, repo_dir / item.name, dirs_exist_ok=True)
    
    # Create training_phases directory with documentation
    phases_dir = repo_dir / "training_phases"
    phases_dir.mkdir(exist_ok=True)
    
    # Copy all markdown files
    for md_file in Path(".").glob("*.md"):
        if md_file.is_file():
            shutil.copy2(md_file, phases_dir / md_file.name)
    
    # Copy test script
    if Path("test_3b_models.py").exists():
        shutil.copy2("test_3b_models.py", phases_dir / "test_3b_models.py")
    
    # Create performance summary
    performance_data = {
        "model_info": {
            "name": "visual-narrator-opt-2.7b",
            "base_model": "facebook/opt-2.7b",
            "parameters": 2700000000,
            "trainable_parameters": 7864320,
            "training_efficiency": 0.0029
        },
        "training_phases": [
            {"phase": "4.1", "description": "2.7B Model Foundation", "status": "completed"},
            {"phase": "4.2", "description": "Dataset Integration", "status": "completed"},
            {"phase": "4.3", "description": "Advanced Training", "status": "completed"},
            {"phase": "4.4", "description": "Benchmarking", "status": "completed"},
            {"phase": "5", "description": "SOTA Optimization", "status": "completed"}
        ],
        "performance_metrics": {
            "targeted_success_rate": 0.88,
            "sota_benchmark_score": 0.577,
            "urban_scenes": 0.75,
            "people_activities": 1.0,
            "nature_scenes": 1.0,
            "average_inference_time": 1.77
        },
        "training_data": {
            "total_examples": 555,
            "urban_scenes": 252,
            "people_activities": 288,
            "nature_scenes": 15,
            "data_quality": "professional_curated"
        }
    }
    
    with open(repo_dir / "training_performance.json", "w") as f:
        json.dump(performance_data, f, indent=2)
    
    print("✅ Hugging Face repository structure created!")
    print(f"📁 Repository location: {repo_dir}")
    print(f"📊 Performance summary: {repo_dir}/training_performance.json")
    print(f"📚 Documentation: {repo_dir}/training_phases/")
    
    return repo_dir

def print_deployment_instructions():
    """Print deployment instructions"""
    print("\n🎯 DEPLOYMENT INSTRUCTIONS:")
    print("1. Make sure you're logged in to Hugging Face:")
    print("   huggingface-cli login")
    print("")
    print("2. Navigate to the repository directory:")
    print("   cd visual-narrator-opt-2.7b")
    print("")
    print("3. Initialize git and push to Hugging Face:")
    print("   git init")
    print("   git add .")
    print("   git commit -m 'Deploy Visual Narrator OPT-2.7B'")
    print("   git push --set-upstream origin main")
    print("")
    print("4. Your model will be available at:")
    print("   https://huggingface.co/your-username/visual-narrator-opt-2.7b")
    print("")
    print("💡 Alternatively, run the deployment script:")
    print("   ./deploy_to_huggingface.sh")

if __name__ == "__main__":
    create_huggingface_repo()
    print_deployment_instructions()
