#!/usr/bin/env python3
"""
Verify Claude model versions and API access
"""

import anthropic
from datetime import datetime

def log(m): print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)

def verify_claude_models():
    """Test Claude API access and available models"""
    
    # Test with our API key
    api_key = os.environ["ANTHROPIC_API_KEY"]
    
    log("🔍 VERIFYING CLAUDE MODEL VERSIONS...")
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        # Test available model versions
        test_models = [
            "claude-3-5-sonnet-20241022",  # What we tried
            "claude-3-5-sonnet-20240620",  # Likely correct
            "claude-3-sonnet-20240229",    # Previous version
            "claude-3-opus-20240229",      # Highest tier
            "claude-3-haiku-20240307"      # Fast tier
        ]
        
        available_models = []
        
        for model in test_models:
            try:
                log(f"  Testing: {model}")
                response = client.messages.create(
                    model=model,
                    max_tokens=50,
                    messages=[{"role": "user", "content": "Say hello in one word"}]
                )
                available_models.append(model)
                log(f"    ✅ AVAILABLE: {model}")
                
            except Exception as e:
                error_msg = str(e)
                if "not_found" in error_msg:
                    log(f"    ❌ NOT FOUND: {model}")
                elif "authentication" in error_msg:
                    log(f"    🔐 AUTH ERROR: {model} - Check API key")
                else:
                    log(f"    ⚠️  OTHER ERROR: {model} - {error_msg[:100]}")
        
        log(f"\n📋 AVAILABLE CLAUDE MODELS:")
        for model in available_models:
            log(f"   • {model}")
            
        if available_models:
            log(f"🎯 RECOMMENDED MODEL: {available_models[0]}")
        else:
            log("❌ NO MODELS AVAILABLE - Check API key or billing")
            
        return available_models
        
    except Exception as e:
        log(f"❌ CLIENT CREATION FAILED: {e}")
        return []

def check_api_quota():
    """Check if we have API quota available"""
    log("\n💰 CHECKING API QUOTA STATUS...")
    
    # Note: Anthropic doesn't have a direct quota endpoint
    # We'll test with a small call
    try:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",  # Use confirmed available model
            max_tokens=10,
            messages=[{"role": "user", "content": "Say 'test'"}]
        )
        
        log("✅ API QUOTA: Available - Test call successful")
        return True
        
    except Exception as e:
        log(f"❌ API QUOTA: Issue detected - {e}")
        return False

if __name__ == "__main__":
    available_models = verify_claude_models()
    quota_ok = check_api_quota()
    
    print(f"\n🎯 SUMMARY:")
    print(f"   • Available Models: {len(available_models)}")
    print(f"   • API Quota: {'✅ OK' if quota_ok else '❌ Issues'}")
    print(f"   • Recommended: Use '{available_models[0] if available_models else 'NO MODEL'}'")
