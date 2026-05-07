#!/bin/bash
API_NAME="visual-narrator-api"
LAMBDA_NAME="visual-narrator-engine"
AWS_REGION="us-east-1"

echo "🌐 Setting up API Gateway..."

# Create REST API
API_ID=$(aws apigateway create-rest-api \
    --name $API_NAME \
    --description "Visual Narrator LLM Engine API" \
    --region $AWS_REGION \
    --query 'id' \
    --output text)

echo "📋 API ID: $API_ID"

# Get root resource ID
ROOT_ID=$(aws apigateway get-resources \
    --rest-api-id $API_ID \
    --region $AWS_REGION \
    --query 'items[0].id' \
    --output text)

# Create proxy resource
aws apigateway create-resource \
    --rest-api-id $API_ID \
    --parent-id $ROOT_ID \
    --path-part "{proxy+}" \
    --region $AWS_REGION

echo "✅ API Gateway setup complete!"
echo "🔗 API URL: https://${API_ID}.execute-api.${AWS_REGION}.amazonaws.com/prod"
