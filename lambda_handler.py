"""
AWS Lambda Handler for Visual Narrator LLM Engine
"""
import json
from mangum import Mangum
from fastapi_server import app

# Create Mangum handler
handler = Mangum(app)

# Optional: Cold start optimization
def lambda_handler(event, context):
    # Add cold start logging
    print("🚀 Visual Narrator Lambda Cold Start")
    return handler(event, context)
