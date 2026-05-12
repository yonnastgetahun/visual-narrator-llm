import anthropic
from datetime import datetime

def log(m): print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)

def test_claude_api():
    """Test Claude API with different configurations"""
    
    # Test different API keys and model names
    test_configs = [
        {
            "name": "Original Key",
            "api_key": os.environ["ANTHROPIC_API_KEY"],
            "models": ["claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20240620", "claude-3-sonnet-20240229"]
        }
    ]
    
    for config in test_configs:
        log(f"🧪 Testing config: {config['name']}")
        
        try:
            client = anthropic.Anthropic(api_key=config['api_key'])
            
            for model in config['models']:
                try:
                    log(f"  Testing model: {model}")
                    response = client.messages.create(
                        model=model,
                        max_tokens=50,
                        messages=[{"role": "user", "content": "Hello"}]
                    )
                    log(f"  ✅ {model}: SUCCESS - {response.content[0].text[:50]}...")
                    
                except Exception as e:
                    log(f"  ❌ {model}: {e}")
                    
        except Exception as e:
            log(f"❌ Client creation failed: {e}")

if __name__ == "__main__":
    test_claude_api()
