#!/usr/bin/env python3
"""
Verify available Claude models with actual API access
"""

import anthropic
from datetime import datetime

def log(m): print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)

def test_claude_models():
    """Test available Claude models with your API key"""
    
    # Your Claude API key
    api_key = "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"
    
    # Available models from your account
    test_models = [
        "claude-3-5-sonnet-20241022",  # Claude Sonnet 4.x (latest)
        "claude-3-opus-20240229",      # Claude Opus 4.x
        "claude-3-sonnet-20240229",    # Claude Sonnet 3.x
        "claude-3-haiku-20240307"      # Claude Haiku 3.x
    ]
    
    log("🔍 TESTING CLAUDE MODEL AVAILABILITY...")
    
    client = anthropic.Anthropic(api_key=api_key)
    
    for model in test_models:
        try:
            log(f"🧪 Testing: {model}")
            response = client.messages.create(
                model=model,
                max_tokens=50,
                messages=[{"role": "user", "content": "Say hello briefly."}]
            )
            log(f"✅ {model}: SUCCESS - '{response.content[0].text}'")
            
        except Exception as e:
            log(f"❌ {model}: FAILED - {e}")
    
    log("🎯 RECOMMENDED MODEL: claude-3-5-sonnet-20241022 (Claude Sonnet 4.x)")

if __name__ == "__main__":
    test_claude_models()
