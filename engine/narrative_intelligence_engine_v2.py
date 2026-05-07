"""
NARRATIVE INTELLIGENCE ENGINE V2
COMPLETE REBUILD with corrected anti-spoiler and audio-visual gap analysis
"""
import json
import subprocess

class NarrativeIntelligenceEngineV2:
    def __init__(self, video_path):
        self.video_path = video_path
        self.scene_duration = self.get_video_duration()
        self.narrative_arc = None
        self.emotional_journey = None
        self.thematic_elements = None
        
    def get_video_duration(self):
        """Get actual video duration"""
        cmd = [
            'ffprobe', '-i', self.video_path,
            '-show_entries', 'format=duration',
            '-v', 'quiet', '-of', 'csv=p=0'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip()) if result.returncode == 0 else 600

    # ===== MACRO-LEVEL ANALYSIS =====
    
    def analyze_narrative_arc(self):
        """SCENE-LEVEL ARC IDENTIFICATION"""
        duration = self.scene_duration
        self.narrative_arc = {
            'genre': 'fantasy_horror',
            'structure': {
                'exposition': (0, duration * 0.15),      # Setup and characters
                'rising_action': (duration * 0.15, duration * 0.4),  # Discovery and tension
                'climax': (duration * 0.4, duration * 0.7),          # White Walker reveal and combat
                'falling_action': (duration * 0.7, duration * 0.9),  # Deaths and aftermath
                'resolution': (duration * 0.9, duration)             # Escape and consequences
            }
        }
        return self.narrative_arc
    
    def analyze_emotional_journey(self):
        """EMOTIONAL JOURNEY: Character emotional states and transitions"""
        self.emotional_journey = {
            'exposition': {'mood': 'apprehensive', 'tension': 'building'},
            'rising_action': {'mood': 'horrified', 'tension': 'high'},
            'climax': {'mood': 'terrified', 'tension': 'peak'},
            'falling_action': {'mood': 'despair', 'tension': 'sustained'},
            'resolution': {'mood': 'panicked', 'tension': 'release'}
        }
        return self.emotional_journey
    
    def analyze_thematic_elements(self):
        """THEMATIC ELEMENTS: Central themes and symbolic content"""
        self.thematic_elements = {
            'central_themes': ['mortality vs immortality', 'summer vs winter', 'courage vs fear'],
            'symbolism': {
                'white_walkers': 'eternal winter and death',
                'night_watch': 'last defense of humanity',
                'haunted_forest': 'unknown ancient powers'
            }
        }
        return self.thematic_elements

    # ===== UPDATED MICRO-LEVEL DECISION TREE =====
    
    def should_narrate_moment_v2(self, moment_time, audio_context, visual_context, narrative_arc):
        """
        UPDATED DECISION TREE V2
        Core question: FOR BLIND VIEWERS, DOES THIS MOMENT NEED AUDIO NARRATION TO FEEL INCLUDED?
        """
        
        # ===== UPDATED JUSTIFICATION CHECK FOR SILENCE =====
        silence_justifications = self.check_silence_justifications(audio_context, visual_context, moment_time, narrative_arc)
        if silence_justifications['should_silence']:
            return {'decision': 'silence', 'reason': silence_justifications['reason']}
        
        # ===== UPDATED NARRATIVE GUARDRAILS =====
        guardrail_result = self.apply_narrative_guardrails_v2(moment_time, visual_context, audio_context, narrative_arc)
        if guardrail_result['decision'] == 'block':
            return {'decision': 'silence', 'reason': guardrail_result['reason']}
        
        # ===== UPDATED NARRATION STRATEGY SELECTION =====
        strategy = self.select_narration_strategy_v2(visual_context, audio_context, moment_time, narrative_arc)
        return {'decision': 'narrate', 'strategy': strategy}
    
    def check_silence_justifications(self, audio_context, visual_context, moment_time, narrative_arc):
        """
        UPDATED: JUSTIFICATION CHECK FOR SILENCE
        Only stay silent when audio successfully conveys the same information
        """
        justifications = {
            'should_silence': False,
            'reason': ''
        }
        
        # CRITICAL FIX: Check if audio conveys COMPLETE narrative information
        audio_sufficiency_checks = [
            # Audio clearly conveys emotional journey
            audio_context.get('clear_emotional_audio', False) and 
            not visual_context.get('critical_visual_information', False),
            
            # Dialogue provides complete context
            audio_context.get('explicit_dialogue_context', False),
            
            # Director intentionally using sound design for specific effect
            audio_context.get('intentional_sound_design', False) and
            not visual_context.get('essential_for_plot', False),
            
            # Audio successfully builds tension without visual help
            audio_context.get('effective_tension_building', False) and
            moment_time < narrative_arc['structure']['climax'][0]
        ]
        
        if any(audio_sufficiency_checks):
            justifications['should_silence'] = True
            justifications['reason'] = 'audio_successfully_conveys_story'
            return justifications
        
        # NEVER stay silent for critical visual information not in audio
        if visual_context.get('critical_visual_information', False) and \
           not audio_context.get('audio_conveys_same_info', False):
            justifications['should_silence'] = False
            justifications['reason'] = 'essential_visual_information_missing_from_audio'
            return justifications
        
        return justifications
    
    def apply_narrative_guardrails_v2(self, moment_time, visual_context, audio_context, narrative_arc):
        """
        UPDATED: APPLY OBJECTIVE NARRATIVE GUARDRAILS
        """
        
        # ===== UPDATED ANTI-SPOILER PROTECTION =====
        spoiler_check = self.anti_spoiler_protection_v2(moment_time, visual_context, audio_context, narrative_arc)
        if spoiler_check['block']:
            return {'decision': 'block', 'reason': spoiler_check['reason']}
        
        # ===== UPDATED STORY-AWARE TIMING ASSESSMENT =====
        timing_check = self.story_aware_timing_v2(moment_time, visual_context, audio_context, narrative_arc)
        if not timing_check['optimal']:
            return {'decision': 'block', 'reason': timing_check['reason']}
        
        return {'decision': 'proceed', 'reason': 'all_guardrails_passed'}
    
    def anti_spoiler_protection_v2(self, moment_time, visual_context, audio_context, narrative_arc):
        """
        UPDATED ANTI-SPOILER PROTECTION
        Only block when BOTH:
        1. Visual contains spoiler information
        2. Audio successfully builds mystery without revealing it
        """
        climax_start = narrative_arc['structure']['climax'][0]
        
        # Check if this is pre-climax and contains supernatural/mystery
        if moment_time < climax_start and visual_context.get('contains_supernatural', False):
            
            # CRITICAL FIX: Only block if audio successfully builds the mystery
            if audio_context.get('successful_mystery_building', False):
                return {
                    'block': True, 
                    'reason': 'audio_successfully_builds_mystery_preserve_director_intent'
                }
            else:
                # Audio doesn't convey the mystery - must narrate to include blind viewers
                return {
                    'block': False,
                    'reason': 'audio_fails_to_convey_mystery_narrate_for_inclusion'
                }
        
        # Check for other spoiler types
        if visual_context.get('foreshadows_death', False) and \
           audio_context.get('subtle_foreshadowing_audio', False):
            return {
                'block': True,
                'reason': 'audio_successfully_foreshadows_preserve_surprise'
            }
        
        return {'block': False, 'reason': 'no_spoiler_concern'}
    
    def story_aware_timing_v2(self, moment_time, visual_context, audio_context, narrative_arc):
        """
        UPDATED STORY-AWARE TIMING ASSESSMENT
        """
        arc_phase = self.get_arc_phase(moment_time, narrative_arc)
        
        # TOO EARLY: Only if director's audio successfully builds anticipation
        if visual_context.get('mystery_element', False) and \
           arc_phase == 'exposition' and \
           audio_context.get('effective_anticipation_building', False):
            return {'optimal': False, 'reason': 'audio_successfully_builds_anticipation_wait_for_reveal'}
        
        # TOO LATE: Information has lost narrative impact AND audio moved on
        if visual_context.get('immediate_visual', False) and \
           moment_time > visual_context.get('optimal_timing', 0) + 10 and \
           audio_context.get('scene_moved_on', False):
            return {'optimal': False, 'reason': 'moment_passed_audio_context_changed'}
        
        # OPTIMAL TIMING: Align with director's revelation timing
        return {'optimal': True, 'reason': 'aligned_with_director_timing_and_audio_context'}
    
    def select_narration_strategy_v2(self, visual_context, audio_context, moment_time, narrative_arc):
        """
        UPDATED NARRATION STRATEGY SELECTION
        """
        
        # CHARACTER INTRODUCTION INTELLIGENCE
        if visual_context.get('first_appearance', False) and \
           not audio_context.get('clear_character_establishment', False):
            return {
                'type': 'character_introduction',
                'approach': 'establish_identity_role_narrative_significance',
                'adjective_density': 'high',  # Rich characterization
                'priority': 'high'  # Essential for understanding
            }
        
        # VISUAL EXPERIENCE TRANSLATION (UPDATED)
        if visual_context.get('visually_striking', False) and \
           not audio_context.get('audio_conveys_visual_impact', False):
            return {
                'type': 'visual_translation', 
                'approach': 'describe_emotional_impact_and_significance',
                'adjective_density': 'high',  # Immersive description
                'priority': 'high'  # Critical for inclusion
            }
        
        # ESSENTIAL PLOT INFORMATION (NEW CATEGORY)
        if visual_context.get('critical_visual_information', False) and \
           not audio_context.get('audio_conveys_same_info', False):
            return {
                'type': 'essential_plot_information',
                'approach': 'clarify_narrative_significance',
                'adjective_density': 'medium',  # Clear and precise
                'priority': 'critical'  # Must not be missed
            }
        
        # EMOTIONAL/MYTHOLOGICAL ENHANCEMENT
        if visual_context.get('emotional_beat', False) and \
           audio_context.get('audio_sets_emotional_tone', False):
            return {
                'type': 'emotional_enhancement',
                'approach': 'add_thematic_resonance_and_depth',
                'adjective_density': 'medium',  # Emotional depth
                'priority': 'medium'  # Enhancement, not essential
            }
        
        # NARRATIVE GUIDANCE
        if (visual_context.get('spatial_shift', False) or 
            visual_context.get('time_passage', False)) and \
           not audio_context.get('audio_conveys_transition', False):
            return {
                'type': 'narrative_guidance',
                'approach': 'orient_and_contextualize',
                'adjective_density': 'low',  # Clear navigation
                'priority': 'medium'
            }
        
        # DEFAULT: Describe visually apparent information
        return {
            'type': 'default_description',
            'approach': 'describe_visually_apparent',
            'adjective_density': 'medium',
            'priority': 'low'
        }
    
    def get_arc_phase(self, moment_time, narrative_arc):
        """Determine which narrative phase the moment belongs to"""
        for phase, (start, end) in narrative_arc['structure'].items():
            if start <= moment_time <= end:
                return phase
        return 'unknown'
    
    # ===== STRATEGIC ASSESSMENT FOR UNCLEAR CASES =====
    
    def strategic_narrative_assessment_v2(self, moment_time, visual_context, audio_context):
        """
        UPDATED STRATEGIC NARRATIVE ASSESSMENT for unclear cases
        """
        assessments = [
            self.enhances_story_comprehension_v2(visual_context, audio_context),
            self.serves_directors_vision_v2(visual_context, audio_context), 
            self.creates_meaningful_inclusion_v2(visual_context, audio_context)
        ]
        
        positive_assessments = sum(assessments)
        
        # If 2+ positive, narrate; otherwise strategic silence
        if positive_assessments >= 2:
            return {'decision': 'narrate', 'reason': f'strategic_enhancement_{positive_assessments}_positive'}
        else:
            return {'decision': 'silence', 'reason': f'strategic_restraint_{positive_assessments}_positive'}
    
    def enhances_story_comprehension_v2(self, visual_context, audio_context):
        """DOES THIS ENHANCE OVERALL STORY COMPREHENSION?"""
        # Narrate if visual contains critical information not in audio
        if visual_context.get('critical_visual_information', False) and \
           not audio_context.get('audio_conveys_same_info', False):
            return True
        
        # Narrate if visual provides essential context missing from audio
        if visual_context.get('provides_essential_context', False) and \
           not audio_context.get('audio_provides_context', False):
            return True
        
        return False
    
    def serves_directors_vision_v2(self, visual_context, audio_context):
        """DOES THIS SERVE THE DIRECTOR'S ARTISTIC VISION?"""
        # Don't narrate if it would disrupt carefully crafted audio experience
        if audio_context.get('intentional_audio_design', False) and \
           not visual_context.get('critical_visual_information', False):
            return False
        
        # Narrate if it enhances the director's visual storytelling
        if visual_context.get('director_visual_focus', False):
            return True
        
        return True  # Default to serving vision through accessibility
    
    def creates_meaningful_inclusion_v2(self, visual_context, audio_context):
        """DOES THIS CREATE MEANINGFUL INCLUSION?"""
        # Always narrate shared viewing moments
        if visual_context.get('shared_viewing_moment', False):
            return True
        
        # Narrate visual jokes, reveals, or emotional peaks
        if visual_context.get('visual_payoff', False) or \
           visual_context.get('emotional_peak', False):
            return True
        
        # Narrate when visual information is key to cultural conversation
        if visual_context.get('culturally_significant_visual', False):
            return True
        
        return False

if __name__ == "__main__":
    # Test the updated engine
    engine = NarrativeIntelligenceEngineV2('gameofthronesseason1episode1.mp4')
    
    # Run macro-level analysis
    print("🎭 UPDATED MACRO-LEVEL ANALYSIS:")
    print(f"Duration: {engine.scene_duration:.2f}s")
    print(f"Narrative Arc: {engine.analyze_narrative_arc()}")
    
    # Test updated decision logic
    print("\n🎯 UPDATED DECISION TREE TEST:")
    
    # Test case: Body discovery (was incorrectly silent)
    test_visual = {
        'critical_visual_information': True,
        'contains_supernatural': True,
        'visually_striking': True,
        'essential_for_plot': True
    }
    test_audio = {
        'audio_conveys_same_info': False,  # Audio only has gasps
        'successful_mystery_building': False,
        'clear_emotional_audio': False
    }
    
    decision = engine.should_narrate_moment_v2(75, test_audio, test_visual, engine.narrative_arc)
    print(f"Body discovery at 75s: {decision}")
    
    # Test case: Atmospheric moment (should be silent)
    test_visual2 = {
        'critical_visual_information': False,
        'visually_striking': False
    }
    test_audio2 = {
        'clear_emotional_audio': True,
        'effective_tension_building': True
    }
    
    decision2 = engine.should_narrate_moment_v2(120, test_audio2, test_visual2, engine.narrative_arc)
    print(f"Atmospheric moment at 120s: {decision2}")
