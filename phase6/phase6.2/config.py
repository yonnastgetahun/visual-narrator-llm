# Phase 6.2 Configuration - OPT-3.5B CONFIRMED
PHASE6_2_CONFIG = {
    "model": {
        "name": "OPT-3.5B",
        "hf_id": "facebook/opt-3.5b",
        "dtype": "float16",
        "architecture": "opt",  # Proven compatible
        "status": "READY"
    },
    "data": {
        "adjective_examples": 995,
        "complex_scenes": 20,
        "total_training": 1015,
        "source": "phase6.2/data/adjective_data/"
    },
    "training": {
        "lora_rank": 32,
        "lora_alpha": 64,
        "learning_rate": 2e-5,
        "batch_size": 2,
        "gradient_accumulation": 4,
        "epochs": 3
    },
    "targets": {
        "adjectives_per_desc": 5.0,  # Improve from 4.15 (OPT-2.7B)
        "sota_benchmark": 0.75,      # Improve from 0.577
        "urban_scenes": 0.95,        # Improve from 0.75
        "training_time": "<6 hours"
    },
    "scaling": {
        "from_model": "OPT-2.7B",
        "parameter_increase": "2.7B → 3.5B (+30%)",
        "expected_improvement": "Higher quality descriptions"
    }
}
