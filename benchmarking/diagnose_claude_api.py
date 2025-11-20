import anthropic
import openai
from datetime import datetime

def log(m): print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)

def diagnose_claude_issue():
    """Diagnose why we can't access highest Claude model"""
    
    # Test with your API key
    api_key = "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"
    
    log("🔍 DIAGNOSING CLAUDE API ACCESS...")
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        log("✅ Claude client created successfully")
        
        # Test different model names
        test_models = [
            "claude-3-5-sonnet-20240620",  # Should be available
            "claude-3-opus-20240229",      # What you're currently using
            "claude-3-sonnet-20240229",    # Alternative
            "claude-3-haiku-20240307"      # Lightweight option
        ]
        
        for model in test_models:
            try:
                log(f"🧪 Testing model: {model}")
                response = client.messages.create(
                    model=model,
                    max_tokens=50,
                    messages=[{"role": "user", "content": "Say hello briefly"}]
                )
                log(f"   ✅ {model}: SUCCESS - {response.content[0].text.strip()}")
                
            except Exception as e:
                log(f"   ❌ {model}: FAILED - {e}")
                
    except Exception as e:
        log(f"❌ Client creation failed: {e}")

def check_api_quota():
    """Check if API quota issues exist"""
    log("\n📊 CHECKING API STATUS...")
    
    # Common issues:
    issues = [
        "API key expired or invalid",
        "Quota exceeded", 
        "Model not available in region",
        "Account needs verification",
        "Billing issues"
    ]
    
    for issue in issues:
        log(f"   • Potential issue: {issue}")

if __name__ == "__main__":
    diagnose_claude_issue()
    check_api_quota()
