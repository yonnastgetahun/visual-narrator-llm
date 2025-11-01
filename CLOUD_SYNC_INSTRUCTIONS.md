# 🔄 Cloud to Local Sync Instructions

## Files Currently on Lambda Cloud (Not Pushed):
- `training/phase3_*.py` - Cloud training scripts
- `cloud/` - Lambda Labs setup scripts  
- `outputs/phase3_*/` - Three trained models
- Various test and verification scripts

## To Sync to Local & HF:

### Option A: Download from Cloud
```bash
# From your LOCAL machine, download cloud files:
scp -r ubuntu@YOUR_LAMBDA_IP:~/visual-narrator-llm/training/phase3_*.py ./
scp -r ubuntu@YOUR_LAMBDA_IP:~/visual-narrator-llm/cloud/ ./
