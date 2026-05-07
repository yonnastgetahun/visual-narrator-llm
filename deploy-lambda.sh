#!/bin/bash
LAMBDA_NAME="visual-narrator-engine"
S3_BUCKET="visual-narrator-models-yg"
S3_KEY="visual-narrator-engine.zip"
AWS_REGION="us-east-1"

echo "🚀 Deploying Visual Narrator Engine to AWS Lambda..."

# Upload to S3 first (make sure package is ready)
aws s3 cp lambda-package/visual-narrator-engine.zip s3://$S3_BUCKET/$S3_KEY

# Check if Lambda exists
aws lambda get-function --function-name $LAMBDA_NAME --region $AWS_REGION > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "📦 Updating existing Lambda function..."
    aws lambda update-function-code \
        --function-name $LAMBDA_NAME \
        --s3-bucket $S3_BUCKET \
        --s3-key $S3_KEY \
        --region $AWS_REGION
else
    echo "❌ Lambda function doesn't exist. You'll need to create it manually via AWS Console first."
    echo "💡 Go to AWS Lambda Console → Create Function → Upload from S3"
    echo "📦 S3 Location: s3://$S3_BUCKET/$S3_KEY"
fi
