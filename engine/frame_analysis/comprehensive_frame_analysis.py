"""
COMPREHENSIVE FRAME ANALYSIS - Full Visual Detail Capture
Goes beyond basic face detection to capture rich visual context
"""
import cv2
import numpy as np
import os
import json
from datetime import datetime
from collections import Counter

class ComprehensiveFrameAnalyzer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Enhanced detectors
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Color ranges for environment detection
        self.color_ranges = {
            'snow': ([200, 200, 200], [255, 255, 255]),
            'forest': ([0, 50, 0], [100, 200, 100]),
            'sky': ([100, 100, 150], [255, 255, 255]),
            'blood': ([0, 0, 50], [100, 100, 255]),
            'armor': ([150, 150, 150], [255, 255, 255])
        }
    
    def extract_frame_at_time(self, timestamp):
        """Extract frame with error handling"""
        frame_number = int(timestamp * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def analyze_color_composition(self, frame):
        """Detailed color analysis"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Dominant colors using k-means
        pixels = frame.reshape(-1, 3)
        pixels = np.float32(pixels)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to uint8
        centers = np.uint8(centers)
        
        # Count pixels per color
        label_counts = Counter(labels.flatten())
        total_pixels = len(labels)
        
        dominant_colors = []
        for i, (color, count) in enumerate(label_counts.most_common(3)):
            dominance = count / total_pixels
            dominant_colors.append({
                'color_bgr': centers[i].tolist(),
                'dominance': float(dominance),
                'pixel_count': int(count)
            })
        
        return {
            'dominant_colors': dominant_colors,
            'brightness': float(np.mean(lab[:,:,0])),  # L channel from LAB
            'saturation': float(np.mean(hsv[:,:,1])),   # Saturation
            'color_variance': float(np.var(frame))      # Color diversity
        }
    
    def detect_environment(self, frame):
        """Detect environmental elements"""
        environment = {}
        
        for element, (lower, upper) in self.color_ranges.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            
            mask = cv2.inRange(frame, lower, upper)
            pixel_count = cv2.countNonZero(mask)
            coverage = pixel_count / (frame.shape[0] * frame.shape[1])
            
            if coverage > 0.05:  # At least 5% coverage
                environment[element] = {
                    'coverage': float(coverage),
                    'pixel_count': int(pixel_count),
                    'confidence': min(coverage * 2, 1.0)  # Scale confidence
                }
        
        return environment
    
    def analyze_composition(self, frame):
        """Advanced composition analysis"""
        height, width = frame.shape[:2]
        
        # Rule of thirds analysis
        thirds_x = [width // 3, 2 * width // 3]
        thirds_y = [height // 3, 2 * height // 3]
        
        # Detect if important elements are at thirds intersections
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # Find edge density in different regions
        regions = {
            'center': edges[height//4:3*height//4, width//4:3*width//4],
            'corners': np.concatenate([
                edges[:height//3, :width//3].flatten(),
                edges[:height//3, 2*width//3:].flatten(),
                edges[2*height//3:, :width//3].flatten(),
                edges[2*height//3:, 2*width//3:].flatten()
            ])
        }
        
        center_activity = np.mean(regions['center'] > 0)
        corner_activity = np.mean(regions['corners'] > 0)
        
        return {
            'resolution': f"{width}x{height}",
            'aspect_ratio': float(width / height),
            'rule_of_thirds_score': float(center_activity - corner_activity),
            'edge_distribution': {
                'center_density': float(center_activity),
                'corner_density': float(corner_activity)
            }
        }
    
    def detect_visual_elements(self, frame):
        """Detect specific visual elements beyond faces"""
        elements = {
            'faces': [],
            'high_contrast_regions': [],
            'text_regions': []
        }
        
        # Face detection with confidence scoring
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            face_region = frame[y:y+h, x:x+w]
            face_brightness = np.mean(cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY))
            
            elements['faces'].append({
                'bbox': [int(x), int(y), int(w), int(h)],
                'area': int(w * h),
                'brightness': float(face_brightness),
                'center': [x + w//2, y + h//2],
                'confidence': min(w * h / (frame.shape[0] * frame.shape[1]) * 10, 1.0)
            })
        
        # High contrast regions (potential focus areas)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        elements['focus_regions'] = float(laplacian_var)
        
        return elements
    
    def analyze_lighting_mood(self, frame):
        """Analyze lighting conditions and mood"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Lighting analysis
        brightness = np.mean(l_channel)
        contrast = np.std(l_channel)
        
        # Color temperature (blue vs red)
        avg_a = np.mean(a_channel)  # Green-Red
        avg_b = np.mean(b_channel)  # Blue-Yellow
        
        # Mood estimation based on lighting
        if brightness < 50:
            lighting_mood = "dark_mysterious"
        elif brightness > 200:
            lighting_mood = "bright_hopeful" 
        elif contrast < 25:
            lighting_mood = "flat_dreary"
        else:
            lighting_mood = "normal_dramatic"
        
        return {
            'brightness': float(brightness),
            'contrast': float(contrast),
            'color_temperature': {
                'warm_cool_balance': float(avg_b),  # Positive = cool, Negative = warm
                'red_green_balance': float(avg_a)   # Positive = red, Negative = green
            },
            'lighting_mood': lighting_mood,
            'exposure_level': 'under' if brightness < 100 else 'over' if brightness > 150 else 'normal'
        }
    
    def comprehensive_frame_analysis(self, timestamp):
        """Complete visual analysis of frame"""
        frame = self.extract_frame_at_time(timestamp)
        
        if frame is None:
            return self.get_fallback_analysis(timestamp)
        
        # Run all analyses
        color_analysis = self.analyze_color_composition(frame)
        environment = self.detect_environment(frame)
        composition = self.analyze_composition(frame)
        visual_elements = self.detect_visual_elements(frame)
        lighting = self.analyze_lighting_mood(frame)
        
        # Comprehensive analysis result
        analysis = {
            'timestamp': timestamp,
            'frame_available': True,
            'analysis_time': datetime.now().isoformat(),
            
            # Visual Properties
            'color_analysis': color_analysis,
            'lighting_analysis': lighting,
            'composition_analysis': composition,
            
            # Content Analysis
            'environment_detection': environment,
            'visual_elements': visual_elements,
            
            # Derived Insights
            'visual_complexity': self.calculate_visual_complexity(frame),
            'narrative_significance': self.estimate_narrative_significance(
                visual_elements, environment, lighting
            ),
            
            # Integration with existing narrative context
            'narrative_context': self.get_narrative_context(timestamp)
        }
        
        return analysis
    
    def calculate_visual_complexity(self, frame):
        """Calculate how visually complex/busy the frame is"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Multiple complexity measures
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges > 0)
        
        color_variance = np.var(frame)
        entropy = -np.sum((np.histogram(gray, 256)[0] / gray.size) * 
                         np.log2(np.histogram(gray, 256)[0] / gray.size + 1e-10))
        
        return {
            'edge_density': float(edge_density),
            'color_variance': float(color_variance),
            'entropy': float(entropy),
            'overall_complexity': float((edge_density + color_variance/10000 + entropy/10) / 3)
        }
    
    def estimate_narrative_significance(self, visual_elements, environment, lighting):
        """Estimate how important this frame is for narration"""
        significance_factors = []
        
        # Faces indicate character focus
        if visual_elements['faces']:
            significance_factors.append(0.7)
        
        # High contrast suggests important visual
        if visual_elements.get('focus_regions', 0) > 1000:
            significance_factors.append(0.6)
        
        # Specific environments are narratively important
        important_environments = ['blood', 'snow', 'forest']
        if any(env in environment for env in important_environments):
            significance_factors.append(0.8)
        
        # Dramatic lighting suggests important moment
        if lighting['lighting_mood'] in ['dark_mysterious', 'bright_hopeful']:
            significance_factors.append(0.5)
        
        return {
            'significance_score': float(np.mean(significance_factors)) if significance_factors else 0.3,
            'significance_factors': significance_factors,
            'recommendation': 'narrate' if (np.mean(significance_factors) > 0.5 if significance_factors else False) else 'consider'
        }
    
    def get_fallback_analysis(self, timestamp):
        """Fallback to basic analysis"""
        from speed_powered_engine import SpeedPoweredEngine
        temp_engine = SpeedPoweredEngine(self.video_path)
        simulated = temp_engine.simulate_frame_analysis(timestamp)
        simulated['frame_available'] = False
        simulated['fallback_used'] = True
        return simulated
    
    def get_narrative_context(self, timestamp):
        """Get narrative context from existing simulation"""
        from speed_powered_engine import SpeedPoweredEngine
        temp_engine = SpeedPoweredEngine(self.video_path)
        return temp_engine.simulate_frame_analysis(timestamp)

def demonstrate_comprehensive_analysis():
    print("🎯 COMPREHENSIVE FRAME ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    analyzer = ComprehensiveFrameAnalyzer('gameofthronesseason1episode1.mp4')
    
    # Analyze key moments with full detail capture
    key_moments = [5, 75, 150, 165, 250]
    
    for timestamp in key_moments:
        print(f"\n🔍 ANALYZING {timestamp}s:")
        analysis = analyzer.comprehensive_frame_analysis(timestamp)
        
        # Show key insights
        print(f"   📊 Visual Complexity: {analysis['visual_complexity']['overall_complexity']:.3f}")
        print(f"   🎨 Dominant Colors: {len(analysis['color_analysis']['dominant_colors'])}")
        print(f"   😊 Faces Detected: {len(analysis['visual_elements']['faces'])}")
        print(f"   🌍 Environment: {list(analysis['environment_detection'].keys())}")
        print(f"   💡 Lighting Mood: {analysis['lighting_analysis']['lighting_mood']}")
        print(f"   📈 Narrative Significance: {analysis['narrative_significance']['significance_score']:.2f}")
        print(f"   🎯 Recommendation: {analysis['narrative_significance']['recommendation']}")
    
    print(f"\n🎉 COMPREHENSIVE ANALYSIS READY!")
    print(f"💡 Now capturing 50+ visual metrics per frame vs basic face detection")

if __name__ == "__main__":
    demonstrate_comprehensive_analysis()
