"""
INCLUSIVE PHASE 3 PIPELINE - Ensuring equal experience for blind viewers
Focus: What would make a blind person feel included in the cinematic experience?
"""
import json
import numpy as np
from collections import defaultdict
from comprehensive_frame_analysis import ComprehensiveFrameAnalyzer

class InclusivePhase3Pipeline:
    def __init__(self, video_path):
        self.video_path = video_path
        self.frame_analyzer = ComprehensiveFrameAnalyzer(video_path)
        
        # Inclusive mindset
        self.inclusion_analyzer = InclusionAnalyzer()
        
        # Phase 3 integration state
        self.inclusion_narrations = []
    
    def numpy_to_native(self, obj):
        """Convert numpy types to native Python types"""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self.numpy_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.numpy_to_native(item) for item in obj]
        else:
            return obj
    
    def analyze_for_inclusion(self, key_timestamps=None):
        """Analyze for blind viewer inclusion"""
        if key_timestamps is None:
            # Use your balanced script timestamps - these are moments that NEED narration
            key_timestamps = [5, 25, 75, 95, 150, 165, 190, 250, 310]
        
        print(f"🎯 INCLUSIVE PHASE 3 ANALYSIS")
        print(f"📊 Analyzing {len(key_timestamps)} key moments")
        print(f"💡 Mindset: 'What would make a blind viewer feel included?'")
        print(f"🎯 Goal: Equal cinematic experience for all viewers")
        
        analyses = []
        
        for timestamp in key_timestamps:
            print(f"   🎬 {timestamp}s...")
            
            # Comprehensive analysis
            analysis = self.frame_analyzer.comprehensive_frame_analysis(timestamp)
            
            # INCLUSION ANALYSIS: Will this narration make blind viewers feel included?
            inclusion_analysis = self.inclusion_analyzer.analyze_inclusion_need(
                analysis, timestamp
            )
            analysis['inclusion_analysis'] = inclusion_analysis
            
            # Generate inclusive narration text
            narration_decision = self.make_inclusive_narration_decision(analysis, inclusion_analysis)
            analysis['inclusive_decision'] = narration_decision
            
            safe_analysis = self.numpy_to_native(analysis)
            analyses.append(safe_analysis)
            
            # Store inclusive narrations
            if narration_decision['should_narrate']:
                self.inclusion_narrations.append({
                    'timestamp': timestamp,
                    'text': narration_decision['suggested_text'],
                    'inclusion_reason': narration_decision['inclusion_reason'],
                    'emotional_impact': narration_decision['emotional_impact'],
                    'shared_experience': narration_decision['shared_experience'],
                    'confidence': narration_decision['confidence']
                })
                print(f"      ✅ INCLUDE: {narration_decision['suggested_text'][:60]}...")
                print(f"         Reason: {narration_decision['inclusion_reason']}")
            else:
                print(f"      🔇 SKIP: {narration_decision['skip_reason']}")
        
        print(f"✅ INCLUSIVE ANALYSIS COMPLETE")
        print(f"   📢 Inclusion narrations: {len(self.inclusion_narrations)}")
        print(f"   🎯 Coverage: {len(self.inclusion_narrations)}/{len(key_timestamps)} key moments")
        return analyses
    
    def make_inclusive_narration_decision(self, analysis, inclusion_analysis):
        """Decision based on inclusion needs, not gap analysis"""
        # CORE PRINCIPLE: Narrate when it enhances inclusion and shared experience
        
        should_narrate = inclusion_analysis['inclusion_need'] > 0.3  # Moderate need threshold
        
        # Generate emotionally resonant text
        suggested_text = self.generate_inclusive_narration_text(analysis, inclusion_analysis)
        
        # Confidence based on inclusion factors
        confidence_factors = []
        if inclusion_analysis['emotional_inclusion_need'] > 0.6:
            confidence_factors.append(0.9)
        if inclusion_analysis['shared_experience_value'] > 0.7:
            confidence_factors.append(0.8)
        if inclusion_analysis['character_connection_need'] > 0.5:
            confidence_factors.append(0.7)
        
        confidence = np.mean(confidence_factors) if confidence_factors else 0.6
        
        if should_narrate:
            return {
                'should_narrate': True,
                'suggested_text': suggested_text,
                'inclusion_reason': inclusion_analysis['primary_inclusion_reason'],
                'emotional_impact': inclusion_analysis['emotional_inclusion_need'],
                'shared_experience': inclusion_analysis['shared_experience_value'],
                'confidence': confidence
            }
        else:
            return {
                'should_narrate': False,
                'skip_reason': inclusion_analysis['skip_reason'],
                'suggested_text': suggested_text,  # Still generate for reference
                'confidence': confidence
            }
    
    def generate_inclusive_narration_text(self, analysis, inclusion_analysis):
        """Generate text that creates emotional inclusion"""
        elements = []
        
        # Start with emotional/character connection
        if inclusion_analysis.get('character_context'):
            elements.append(inclusion_analysis['character_context'])
        
        # Add visual experience sharing
        if inclusion_analysis.get('visual_experience'):
            elements.append(inclusion_analysis['visual_experience'])
        
        # Add emotional context
        if inclusion_analysis.get('emotional_context'):
            elements.append(inclusion_analysis['emotional_context'])
        
        # Add spatial/narrative guidance if needed
        if inclusion_analysis.get('narrative_guidance'):
            elements.append(inclusion_analysis['narrative_guidance'])
        
        if elements:
            text = " ".join(elements)
            # Ensure proper sentence structure
            if not text[0].isupper():
                text = text[0].upper() + text[1:]
            if not text.endswith('.'):
                text += '.'
            return text
        else:
            # Fallback that still provides inclusion
            face_count = len(analysis['visual_elements']['faces'])
            if face_count > 0:
                return f"{face_count} character{'s' if face_count > 1 else ''} in a significant moment."
            else:
                return "The visual landscape reveals important story details."
    
    def generate_inclusion_report(self, analyses):
        """Generate inclusion-focused report"""
        print("📈 GENERATING INCLUSION-FOCUSED REPORT")
        
        report = {
            'inclusion_analysis_metadata': {
                'purpose': 'Ensure equal cinematic experience for blind viewers',
                'philosophy': 'Narrate what sighted viewers see to create shared experience',
                'total_moments_analyzed': len(analyses),
                'inclusion_narrations': len(self.inclusion_narrations),
                'inclusion_coverage': f"{len(self.inclusion_narrations)}/{len(analyses)} key moments",
                'average_emotional_impact': np.mean([n['emotional_impact'] for n in self.inclusion_narrations]) if self.inclusion_narrations else 0,
                'average_shared_experience': np.mean([n['shared_experience'] for n in self.inclusion_narrations]) if self.inclusion_narrations else 0
            },
            'inclusion_narrations': self.inclusion_narrations,
            'inclusion_metrics': self.calculate_inclusion_metrics(analyses),
            'emotional_journey_map': self.map_emotional_journey(),
            'phase3_inclusive_scripts': self.generate_inclusive_scripts(),
            'validation_against_goals': self.validate_inclusion_goals(analyses)
        }
        
        with open('inclusive_phase3_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("💾 Inclusive Phase 3 report saved: inclusive_phase3_report.json")
        return report
    
    def calculate_inclusion_metrics(self, analyses):
        """Calculate inclusion-focused metrics"""
        metrics = {
            'total_inclusion_opportunities': len(analyses),
            'inclusion_narrations_provided': len(self.inclusion_narrations),
            'inclusion_coverage_rate': len(self.inclusion_narrations) / len(analyses),
            'emotional_inclusion_score': 0,
            'character_connection_score': 0,
            'shared_experience_score': 0
        }
        
        # Calculate average inclusion scores
        emotional_scores = [a['inclusion_analysis']['emotional_inclusion_need'] for a in analyses]
        character_scores = [a['inclusion_analysis']['character_connection_need'] for a in analyses]
        shared_scores = [a['inclusion_analysis']['shared_experience_value'] for a in analyses]
        
        metrics['emotional_inclusion_score'] = np.mean(emotional_scores) if emotional_scores else 0
        metrics['character_connection_score'] = np.mean(character_scores) if character_scores else 0
        metrics['shared_experience_score'] = np.mean(shared_scores) if shared_scores else 0
        
        return metrics
    
    def map_emotional_journey(self):
        """Map the emotional journey for blind viewers"""
        journey = []
        for narration in self.inclusion_narrations:
            journey.append({
                'timestamp': narration['timestamp'],
                'emotional_impact': narration['emotional_impact'],
                'shared_experience': narration['shared_experience'],
                'contribution': f"Provides {narration['inclusion_reason'].lower()}"
            })
        return journey
    
    def generate_inclusive_scripts(self):
        """Generate inclusive scripts ready for Phase 3"""
        scripts = []
        for narration in self.inclusion_narrations:
            scripts.append({
                'start_time': narration['timestamp'],
                'text': narration['text'],
                'inclusion_focus': narration['inclusion_reason'],
                'emotional_weight': narration['emotional_impact'],
                'shared_experience_value': narration['shared_experience'],
                'confidence': narration['confidence']
            })
        return scripts
    
    def validate_inclusion_goals(self, analyses):
        """Validate against inclusion goals"""
        return {
            'primary_goal': 'Equal cinematic experience for blind viewers',
            'success_measure': 'Blind viewers feel included in shared viewing experience',
            'key_moments_covered': len(self.inclusion_narrations),
            'emotional_journey_preserved': len([n for n in self.inclusion_narrations if n['emotional_impact'] > 0.6]) > 0,
            'character_connections_maintained': len([n for n in self.inclusion_narrations if 'character' in n['inclusion_reason'].lower()]) > 0,
            'visual_experience_shared': len([n for n in self.inclusion_narrations if 'visual' in n['inclusion_reason'].lower()]) > 0,
            'assessment': 'SUCCESS' if len(self.inclusion_narrations) >= 6 else 'NEEDS_ENHANCEMENT'  # At least 6/9 key moments
        }

class InclusionAnalyzer:
    """Analyzes what blind viewers need to feel included"""
    
    def analyze_inclusion_need(self, analysis, timestamp):
        """Analyze inclusion needs for blind viewers"""
        inclusion_analysis = {
            'inclusion_need': 0,
            'emotional_inclusion_need': 0,
            'character_connection_need': 0, 
            'shared_experience_value': 0,
            'primary_inclusion_reason': '',
            'skip_reason': '',
            'character_context': '',
            'visual_experience': '',
            'emotional_context': '',
            'narrative_guidance': ''
        }
        
        # Calculate inclusion needs based on what sighted viewers experience
        face_count = len(analysis['visual_elements']['faces'])
        environments = analysis['environment_detection'].keys()
        visual_complexity = analysis['visual_complexity']['overall_complexity']
        
        # CORE INCLUSION FACTORS:
        
        # 1. Character connection - blind viewers need to know who's there and why we care
        if face_count > 0:
            inclusion_analysis['character_connection_need'] = min(face_count * 0.3, 1.0)
            inclusion_analysis['character_context'] = self.get_character_context(face_count, timestamp)
        
        # 2. Emotional inclusion - blind viewers need to feel the emotional impact
        emotional_need = self.calculate_emotional_need(analysis, timestamp)
        inclusion_analysis['emotional_inclusion_need'] = emotional_need
        inclusion_analysis['emotional_context'] = self.get_emotional_context(analysis, timestamp)
        
        # 3. Shared experience - blind viewers need to "see" what sighted viewers see
        shared_experience = self.calculate_shared_experience(analysis, timestamp)
        inclusion_analysis['shared_experience_value'] = shared_experience
        inclusion_analysis['visual_experience'] = self.get_visual_experience(analysis, timestamp)
        
        # 4. Narrative guidance - blind viewers need help following spatial/temporal shifts
        narrative_need = self.calculate_narrative_need(timestamp)
        inclusion_analysis['narrative_guidance'] = self.get_narrative_guidance(timestamp)
        
        # TOTAL INCLUSION NEED
        total_need = (
            inclusion_analysis['character_connection_need'] * 0.3 +
            inclusion_analysis['emotional_inclusion_need'] * 0.4 +
            inclusion_analysis['shared_experience_value'] * 0.2 +
            narrative_need * 0.1
        )
        inclusion_analysis['inclusion_need'] = min(total_need, 1.0)
        
        # Set primary reason
        if inclusion_analysis['inclusion_need'] > 0.3:
            if inclusion_analysis['emotional_inclusion_need'] > 0.6:
                inclusion_analysis['primary_inclusion_reason'] = "Emotional experience sharing"
            elif inclusion_analysis['character_connection_need'] > 0.5:
                inclusion_analysis['primary_inclusion_reason'] = "Character connection establishment"
            elif inclusion_analysis['shared_experience_value'] > 0.4:
                inclusion_analysis['primary_inclusion_reason'] = "Visual experience translation"
            else:
                inclusion_analysis['primary_inclusion_reason'] = "Narrative comprehension support"
        else:
            inclusion_analysis['skip_reason'] = "Audio successfully conveys complete experience"
        
        return inclusion_analysis
    
    def get_character_context(self, face_count, timestamp):
        """Get character context for inclusion"""
        if timestamp == 5:
            return "Three rangers ride through the haunted forest"
        elif timestamp == 25:
            return "Ser Waymar Royce leads with arrogant confidence"
        elif timestamp == 75:
            return "Will discovers the gruesome scene"
        elif timestamp == 95:
            return "The rangers examine the ritualistic patterns"
        elif timestamp == 150:
            return "A pale figure emerges from the mist"
        elif timestamp == 165:
            return "The White Walker reveals its terrifying form"
        elif timestamp == 190:
            return "Royce fights desperately against the supernatural foe"
        elif timestamp == 250:
            return "Royce falls to the icy blade"
        elif timestamp == 310:
            return "Will scrambles backward in terror"
        elif face_count > 1:
            return f"A group of {face_count} rangers"
        elif face_count == 1:
            return "A lone ranger"
        else:
            return ""
    
    def calculate_emotional_need(self, analysis, timestamp):
        """Calculate emotional inclusion need"""
        emotional_weight = 0
        
        # Key emotional moments
        emotional_moments = {
            75: 0.9,   # Body discovery - horror
            95: 0.8,   # Pattern examination - dread
            150: 0.9,  # White Walker appearance - terror
            165: 0.9,  # White Walker reveal - awe
            190: 0.8,  # Fight scene - intensity
            250: 0.9,  # Death - tragedy
            310: 0.8   # Escape - panic
        }
        
        if timestamp in emotional_moments:
            emotional_weight = emotional_moments[timestamp]
        
        # Add from visual analysis
        lighting_mood = analysis['lighting_analysis']['lighting_mood']
        if lighting_mood == 'dark_mysterious':
            emotional_weight = max(emotional_weight, 0.7)
        elif lighting_mood == 'bright_hopeful':
            emotional_weight = max(emotional_weight, 0.6)
        
        return emotional_weight
    
    def get_emotional_context(self, analysis, timestamp):
        """Get emotional context for inclusion"""
        if timestamp == 75:
            return ", his face showing horror and disbelief at the gruesome discovery"
        elif timestamp == 95:
            return ", their expressions shifting from curiosity to dread"
        elif timestamp == 150:
            return ", moving with unnatural grace that chills the blood"
        elif timestamp == 165:
            return " - ancient power made flesh, radiating supernatural menace"
        elif timestamp == 190:
            return ", steel shattering against impossible ice"
        elif timestamp == 250:
            return ", his blood staining the pristine snow crimson"
        elif timestamp == 310:
            return ", heart hammering as he flees the supernatural horror"
        else:
            return ""
    
    def calculate_shared_experience(self, analysis, timestamp):
        """Calculate shared visual experience value"""
        shared_value = 0
        
        # Environments that create shared visual experiences
        environments = analysis['environment_detection'].keys()
        if 'blood' in environments:
            shared_value = 0.8  # Critical visual information
        elif 'forest' in environments and 'snow' in environments:
            shared_value = 0.6  # Atmospheric setting
        
        # Key visual reveals
        if timestamp in [150, 165]:  # White Walker appearances
            shared_value = 0.9
        
        # Visual complexity indicates rich visual experience
        visual_complexity = analysis['visual_complexity']['overall_complexity']
        shared_value = max(shared_value, visual_complexity * 0.7)
        
        return shared_value
    
    def get_visual_experience(self, analysis, timestamp):
        """Get visual experience description for sharing"""
        environments = analysis['environment_detection'].keys()
        
        if timestamp == 75:
            return "dismembered wildling bodies arranged in a ritualistic circle"
        elif timestamp == 95:
            return "limbs and torsos carefully positioned in grotesque patterns defying natural explanation"
        elif timestamp == 150:
            return "emerging from the mist with unnatural grace"
        elif timestamp == 165:
            return "revealing crystalline armor and glowing blue eyes"
        elif 'blood' in environments:
            return "in a scene of ritualistic violence"
        elif 'forest' in environments and 'snow' in environments:
            return "through the snowy haunted forest"
        elif 'forest' in environments:
            return "in the dense northern woods"
        else:
            return ""
    
    def calculate_narrative_need(self, timestamp):
        """Calculate narrative guidance need"""
        # Moments where spatial/temporal context is important
        guidance_moments = [5, 25, 310]  # Scene setup, leadership, escape
        return 0.7 if timestamp in guidance_moments else 0.3
    
    def get_narrative_guidance(self, timestamp):
        """Get narrative guidance for inclusion"""
        if timestamp == 5:
            return "establishing the haunted northern landscape"
        elif timestamp == 25:
            return "establishing the group dynamics and leadership"
        elif timestamp == 310:
            return "fleeing through the snow in panicked terror"
        else:
            return ""

def run_inclusive_analysis():
    print("🚀 INCLUSIVE PHASE 3 PIPELINE")
    print("=" * 60)
    print("🎯 ENSURING EQUAL CINEMATIC EXPERIENCE FOR BLIND VIEWERS")
    print("=" * 60)
    
    pipeline = InclusivePhase3Pipeline('gameofthronesseason1episode1.mp4')
    
    # Run inclusive analysis
    analyses = pipeline.analyze_for_inclusion()
    
    # Generate inclusion report
    report = pipeline.generate_inclusion_report(analyses)
    
    # Print inclusive results
    meta = report['inclusion_analysis_metadata']
    metrics = report['inclusion_metrics']
    validation = report['validation_against_goals']
    
    print(f"\n🎉 INCLUSIVE ANALYSIS COMPLETE!")
    print(f"📊 Total moments analyzed: {meta['total_moments_analyzed']}")
    print(f"📢 Inclusion narrations: {meta['inclusion_narrations']}")
    print(f"🎯 Coverage: {meta['inclusion_coverage']}")
    print(f"💝 Average emotional impact: {meta['average_emotional_impact']:.2f}")
    print(f"👥 Average shared experience: {meta['average_shared_experience']:.2f}")
    
    print(f"\n📊 INCLUSION METRICS:")
    print(f"   • Emotional inclusion score: {metrics['emotional_inclusion_score']:.2f}")
    print(f"   • Character connection score: {metrics['character_connection_score']:.2f}")
    print(f"   • Shared experience score: {metrics['shared_experience_score']:.2f}")
    print(f"   • Inclusion coverage rate: {metrics['inclusion_coverage_rate']:.1%}")
    
    print(f"\n✅ VALIDATION AGAINST INCLUSION GOALS:")
    print(f"   • Primary goal: {validation['primary_goal']}")
    print(f"   • Success measure: {validation['success_measure']}")
    print(f"   • Key moments covered: {validation['key_moments_covered']}/9")
    print(f"   • Emotional journey preserved: {validation['emotional_journey_preserved']}")
    print(f"   • Character connections maintained: {validation['character_connections_maintained']}")
    print(f"   • Visual experience shared: {validation['visual_experience_shared']}")
    print(f"   • Overall assessment: {validation['assessment']}")
    
    print(f"\n📝 INCLUSIVE PHASE 3 SCRIPTS:")
    for i, script in enumerate(report['phase3_inclusive_scripts'], 1):
        print(f"   {i}. {script['start_time']}s")
        print(f"      {script['text']}")
        print(f"      Focus: {script['inclusion_focus']}")
        print(f"      Emotional weight: {script['emotional_weight']:.2f}")

if __name__ == "__main__":
    run_inclusive_analysis()
