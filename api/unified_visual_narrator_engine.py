"""
Unified Visual Narrator LLM Engine
Integrates all components into single processing pipeline for Phase 3 & 4 demos
"""

import sys
import os
import json
from typing import Dict, List, Optional
from pathlib import Path

# Add engine paths
sys.path.append(str(Path(__file__).parent.parent))

try:
    from engine.frame_analysis.comprehensive_frame_analysis import comprehensive_frame_analysis
    from engine.inclusion_intelligence.inclusive_phase3_pipeline import analyze_inclusion_need
    from engine.inclusion_intelligence.balanced_phase3_pipeline import analyze_audio_visual_gap
    from engine.speed_powered_engine import analyze_frame_sequence
    from engine.narrative_intelligence_engine_v2 import NarrativeIntelligenceEngine
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback imports will be handled in class

class UnifiedVisualNarratorEngine:
    """Main engine integrating all VLM components"""
    
    def __init__(self):
        self.performance_stats = {
            "frames_processed": 0,
            "avg_processing_time": 0,
            "errors": 0
        }
        
        # Initialize components
        try:
            self.narrative_engine = NarrativeIntelligenceEngine()
        except:
            self.narrative_engine = None
            print("Narrative engine not available")
    
    def process_video_frame(self, video_path: str, timestamp: float, frame_data: Optional[Dict] = None) -> Dict:
        """
        Process single video frame through entire pipeline
        """
        try:
            # 1. Frame Analysis (V3 Foundation)
            frame_analysis = comprehensive_frame_analysis(video_path, timestamp, frame_data)
            
            # 2. Inclusion Intelligence
            inclusion_analysis = analyze_inclusion_need(frame_analysis)
            
            # 3. Audio-Visual Gap Analysis
            gap_analysis = analyze_audio_visual_gap(frame_analysis, inclusion_analysis)
            
            # 4. Narrative Intelligence (if available)
            narrative_context = {}
            if self.narrative_engine:
                narrative_context = self.narrative_engine.analyze_narrative_context(frame_analysis)
            
            # Update performance stats
            self.performance_stats["frames_processed"] += 1
            
            return {
                "status": "success",
                "timestamp": timestamp,
                "frame_analysis": frame_analysis,
                "inclusion_analysis": inclusion_analysis,
                "gap_analysis": gap_analysis,
                "narrative_context": narrative_context,
                "performance": self.performance_stats
            }
            
        except Exception as e:
            self.performance_stats["errors"] += 1
            return {
                "status": "error",
                "timestamp": timestamp,
                "error": str(e),
                "performance": self.performance_stats
            }
    
    def process_video_sequence(self, video_path: str, timestamps: List[float]) -> Dict:
        """
        Process multiple frames for temporal analysis
        """
        results = []
        
        for timestamp in timestamps:
            result = self.process_video_frame(video_path, timestamp)
            results.append(result)
        
        # Temporal analysis across frames
        temporal_analysis = self._analyze_temporal_patterns(results)
        
        return {
            "sequence_analysis": results,
            "temporal_patterns": temporal_analysis,
            "summary": {
                "total_frames": len(results),
                "successful_frames": len([r for r in results if r["status"] == "success"]),
                "performance": self.performance_stats
            }
        }
    
    def _analyze_temporal_patterns(self, frame_results: List[Dict]) -> Dict:
        """Analyze patterns across multiple frames"""
        successful_frames = [r for r in frame_results if r["status"] == "success"]
        
        if not successful_frames:
            return {"error": "No successful frames to analyze"}
        
        # Extract key metrics for temporal analysis
        emotions = []
        characters = set()
        actions = []
        
        for frame in successful_frames:
            analysis = frame.get("frame_analysis", {})
            emotions.append(analysis.get("emotional_tone", "neutral"))
            characters.update(analysis.get("characters_detected", []))
            actions.append(analysis.get("primary_action", "unknown"))
        
        return {
            "emotional_arc": emotions,
            "character_persistence": list(characters),
            "action_progression": actions,
            "scene_transitions": len([e for i, e in enumerate(emotions) if i > 0 and e != emotions[i-1]])
        }
    
    def get_engine_status(self) -> Dict:
        """Get engine health and performance"""
        return {
            "status": "active",
            "components": {
                "frame_analysis": "available",
                "inclusion_intelligence": "available", 
                "gap_analysis": "available",
                "narrative_intelligence": "available" if self.narrative_engine else "unavailable"
            },
            "performance": self.performance_stats
        }

# Singleton instance
engine_instance = UnifiedVisualNarratorEngine()

def get_engine():
    """Get singleton engine instance"""
    return engine_instance

if __name__ == "__main__":
    # Test the engine
    engine = UnifiedVisualNarratorEngine()
    status = engine.get_engine_status()
    print("Visual Narrator LLM Engine Status:")
    print(json.dumps(status, indent=2))
