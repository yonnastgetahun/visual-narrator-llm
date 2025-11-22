import re
import os

# Files that contain API keys (from the error message)
files_to_clean = [
    "benchmarking/benchmark_adjective_dominance_highest.py",
    "benchmarking/benchmark_sota_comparison.py", 
    "benchmarking/diagnose_claude_api.py",
    "benchmarking/real_claude_benchmark.py",
    "benchmarking/verify_claude_versions.py",
    "benchmarking/benchmark_professional_final.py",
    "benchmarking/benchmark_real_scores.py",
    "benchmarking/benchmark_real_scores_clean.py",
    "benchmarking/benchmark_final_comprehensive.py"
]

# Patterns to replace API keys
patterns = {
    r"ANTHROPIC_API_KEY\s*=\s*['\"][^'\"]+['\"]": "ANTHROPIC_API_KEY = 'your_anthropic_api_key_here'",
    r"OPENAI_API_KEY\s*=\s*['\"][^'\"]+['\"]": "OPENAI_API_KEY = 'your_openai_api_key_here'",
    r"claude_api_key\s*=\s*['\"][^'\"]+['\"]": "claude_api_key = 'your_claude_api_key_here'",
    r"openai.api_key\s*=\s*['\"][^'\"]+['\"]": "openai.api_key = 'your_openai_api_key_here'"
}

for file_path in files_to_clean:
    if os.path.exists(file_path):
        print(f"Cleaning {file_path}...")
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Replace actual API keys with placeholders
        for pattern, replacement in patterns.items():
            content = re.sub(pattern, replacement, content)
        
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"✅ Cleaned {file_path}")

print("🎉 All files cleaned. API keys replaced with placeholders.")
