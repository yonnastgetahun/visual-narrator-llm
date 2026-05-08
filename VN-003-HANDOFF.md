# VN-003 Handoff

## Status

Completed on 2026-05-08.

The `visual-narrator-engine` Lambda now uses AWS Rekognition for object detection and has been redeployed successfully.

## What Changed

- Attached `AmazonRekognitionReadOnlyAccess` to `lambda-execution-role`.
- Replaced mocked object output in the deployed Lambda path with:
  - `rekognition.detect_labels`
  - `MaxLabels=10`
  - `MinConfidence=70.0`
- Added:
  - `objects_detected`
  - `object_count`
  - `detection_source`
- Added graceful fallback behavior:
  - `objects_detected: []`
  - `detection_source: "fallback"`
- Updated the `live-metrics` handler path to return `object_detection: "rekognition"`.

## Deployment Facts

- Lambda function:
  - `visual-narrator-engine`
- Lambda package type:
  - `Image`
- Active resolved image:
  - `092439868784.dkr.ecr.us-east-1.amazonaws.com/vn-engine-lambda@sha256:287113e8588f839e73908f37a00448a24bc0cc18b135331b9fd4548c2659d708`
- ECR tag updated:
  - `092439868784.dkr.ecr.us-east-1.amazonaws.com/vn-engine-lambda:latest`

## Verification

Direct Lambda invoke with a real JPEG returned:

```json
{
  "analysis": {
    "objects_detected": [
      { "label": "Abyssinian", "confidence": 99.6 },
      { "label": "Animal", "confidence": 99.6 },
      { "label": "Cat", "confidence": 99.6 },
      { "label": "Mammal", "confidence": 99.6 },
      { "label": "Pet", "confidence": 99.6 },
      { "label": "Manx", "confidence": 95.9 }
    ],
    "object_count": 6,
    "detection_source": "rekognition"
  }
}
```

`live-metrics` Lambda HTTP event returned:

```json
{
  "object_detection": "rekognition",
  "detection_source": "rekognition",
  "status": "active"
}
```

## Local Files Touched

- `api/fastapi_server.py`
- `engine/frame_analysis/comprehensive_frame_analysis.py`
- `vn003-codebuild-trust-policy.json`
- `vn003-codebuild-policy.json`
- `vn003-codebuild-project.json`

## AWS Build Workaround

Local Docker was unhealthy, so deployment was completed through AWS CodeBuild.

Created for this task:

- IAM role:
  - `vn-codebuild-service-role`
- CodeBuild project:
  - `vn003-vn-engine-image-build`
- S3 source bundle:
  - `s3://visual-narrator-models-yg/vn003/vn-engine-image.zip`

## Notes

- `package-lock.json.backup` was already untracked and is unrelated.
- The temporary remote build artifacts can be kept for auditability or removed later if they are no longer needed.
