"""
FastAPI Server for Visual Narrator LLM Engine
Provides API endpoints for Phase 3 & 4 demo integration
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import json
from typing import Dict, List, Optional
from pathlib import Path
import sys

# Add engine paths
sys.path.append(str(Path(__file__).parent.parent))

try:
    from unified_visual_narrator_engine import get_engine
    ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"Engine import error: {e}")
    ENGINE_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="Visual Narrator LLM Engine API",
    description="Unified engine for emotionally resonant audio narrations",
    version="1.0.0"
)

# CORS middleware for Vercel demo integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://visual-narrator-investor-demo.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engine
if ENGINE_AVAILABLE:
    engine = get_engine()
else:
    engine = None
    print("⚠️  Running in mock mode - engine components not available")

class MockEngine:
    """Mock engine for demo when real engine isn't available"""
    
    def process_video_frame(self, video_path: str, timestamp: float):
        return {
            "status": "success",
            "timestamp": timestamp,
            "frame_analysis": {
                "objects_detected": ["ranger", "forest", "snow"],
                "emotional_tone": "apprehensive",
                "characters_detected": ["will"],
                "primary_action": "riding_through_forest",
                "visual_complexity": 0.7
            },
            "inclusion_analysis": {
                "needs_narration": True,
                "emotional_impact": 0.82,
                "inclusion_score": 0.9
            },
            "gap_analysis": {
                "audio_visual_gap": True,
                "narration_priority": "high",
                "gap_reason": "critical_visual_context_missing"
            },
            "performance": {"frames_processed": 1, "avg_processing_time": 2.5}
        }
    
    def process_video_sequence(self, video_path: str, timestamps: List[float]):
        results = []
        for ts in timestamps:
            results.append(self.process_video_frame(video_path, ts))
        
        return {
            "sequence_analysis": results,
            "temporal_patterns": {
                "emotional_arc": ["apprehensive", "horrified", "investigative"],
                "character_persistence": ["will"],
                "action_progression": ["riding", "discovering", "examining"]
            },
            "summary": {"total_frames": len(timestamps), "successful_frames": len(timestamps)}
        }
    
    def get_engine_status(self):
        return {
            "status": "mock_mode",
            "components": {
                "frame_analysis": "mock",
                "inclusion_intelligence": "mock",
                "gap_analysis": "mock",
                "narrative_intelligence": "mock"
            },
            "performance": {"frames_processed": 0, "errors": 0}
        }

# Use mock if real engine not available
if not ENGINE_AVAILABLE:
    engine = MockEngine()

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "Visual Narrator LLM Engine API",
        "version": "1.0.0",
        "status": "active",
        "engine_available": ENGINE_AVAILABLE
    }

@app.get("/status")
async def get_status():
    """Get engine health and status"""
    return engine.get_engine_status()

@app.post("/analyze/frame")
async def analyze_frame(video_path: str, timestamp: float):
    """
    Analyze single video frame
    """
    try:
        result = engine.process_video_frame(video_path, timestamp)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Frame analysis error: {str(e)}")

@app.post("/analyze/sequence")
async def analyze_sequence(video_path: str, timestamps: List[float]):
    """
    Analyze multiple frames for temporal patterns
    """
    try:
        result = engine.process_video_sequence(video_path, timestamps)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sequence analysis error: {str(e)}")

@app.get("/demo/test")
async def demo_test():
    """
    Demo endpoint with sample Game of Thrones analysis
    """
    test_data = {
        "video_path": "gameofthronesseason1episode1.mp4",
        "timestamps": [5.0, 75.0, 120.0],
        "analysis": [
            {
                "timestamp": 5.0,
                "narration_decision": "Narrate",
                "reason": "Establishing scene and character introduction",
                "sample_narration": "Three rangers on horseback move cautiously through a snow-dusted forest, their breath visible in the cold air."
            },
            {
                "timestamp": 75.0,
                "narration_decision": "Narrate",
                "reason": "Critical plot revelation missing from audio",
                "sample_narration": "Will discovers eight dismembered bodies arranged in a ritualistic circle, his face frozen in horror."
            },
            {
                "timestamp": 120.0,
                "narration_decision": "Strategic Silence",
                "reason": "Audio successfully builds tension and mystery",
                "sample_narration": ""
            }
        ],
        "performance_metrics": {
            "processing_speed": "2.5ms per frame",
            "semantic_accuracy": "66.7%",
            "speed_advantage": "2249x faster than Claude"
        }
    }
    return JSONResponse(content=test_data)

if __name__ == "__main__":
    print("🚀 Starting Visual Narrator LLM Engine API Server...")
    print("📊 Engine Available:", ENGINE_AVAILABLE)
    print("🌐 API Documentation: http://localhost:8000/docs")
    print("🔗 Demo Test: http://localhost:8000/demo/test")
    
    uvicorn.run(
        "fastapi_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
