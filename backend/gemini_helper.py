"""
Gemini Vision API Integration for Medical Image Analysis
This module provides enhanced fracture detection and localization using Google's Gemini API
"""
import google.generativeai as genai
import os
import json
import re
from PIL import Image
import numpy as np

class GeminiAnalyzer:
    def __init__(self, api_key=None):
        """
        Initialize Gemini API client
        Args:
            api_key: Google AI API key. If None, will try to load from environment variable
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Please set it in .env file or pass it to constructor")
        
        genai.configure(api_key=self.api_key)
        
        # List available models to find the correct one
        try:
            print("Checking available Gemini models...")
            available_models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)
                    print(f"  Available: {m.name}")
            
            if not available_models:
                raise ValueError("No models support generateContent")
            
            # Prioritize models for vision/medical imaging
            # Prefer: flash (fast), 2.0/2.5 (latest), pro (high quality)
            preference_order = [
                'gemini-2.5-flash',
                'gemini-2.5-pro', 
                'gemini-2.0-flash',
                'gemini-flash-latest',
                'gemini-pro-latest',
                'gemini-2.0-pro-exp'
            ]
            
            model_to_use = None
            
            # Try preferred models first
            for preferred in preference_order:
                for available in available_models:
                    if preferred in available.lower():
                        model_to_use = available
                        break
                if model_to_use:
                    break
            
            # If no preferred model found, use the first available
            if not model_to_use:
                model_to_use = available_models[0]
            
            # Extract just the model name (remove 'models/' prefix if present)
            if model_to_use.startswith('models/'):
                model_to_use = model_to_use.replace('models/', '')
            
            print(f"✓ Using Gemini model: {model_to_use}")
            self.model = genai.GenerativeModel(model_to_use)
            
        except Exception as e:
            print(f"Error listing models: {e}")
            # Fallback to a known working model
            print("Falling back to gemini-pro")
            self.model = genai.GenerativeModel('gemini-pro')
        
    def analyze_xray(self, image_path, body_part_hint=None):
        """
        Analyze X-ray image for fractures using Gemini Vision API
        
        Args:
            image_path: Path to the X-ray image
            body_part_hint: Optional hint about the body part (from your existing model)
            
        Returns:
            dict with fracture analysis results
        """
        try:
            # Load image
            img = Image.open(image_path)
            
            # Create a detailed prompt for medical image analysis
            prompt = self._create_analysis_prompt(body_part_hint)
            
            # Generate content with image
            response = self.model.generate_content([prompt, img])
            
            # Parse the response
            analysis = self._parse_response(response.text)
            
            return analysis
            
        except Exception as e:
            print(f"Error in Gemini analysis: {str(e)}")
            return self._get_fallback_response()
    
    def _create_analysis_prompt(self, body_part_hint=None):
        """
        Create a simplified prompt for fracture analysis
        """
        base_prompt = """You are an expert radiologist analyzing an X-ray for bone fractures.

Analyze this X-ray and provide ONLY a JSON response with this exact format:

{
    "fracture_detected": true/false,
    "confidence": 0.0-1.0,
    "body_part": "simple bone name (e.g., 'tibia', 'femur', 'radius', 'humerus', 'fibula', 'ulna', 'clavicle', 'rib')",
    "fracture_type": "simple type if fracture detected (e.g., 'hairline', 'complete', 'compound')",
    "location": {
        "anatomical_region": "simple location (e.g., 'mid-shaft', 'distal end', 'proximal end')",
        "side": "left/right/center",
        "position": "upper/middle/lower"
    },
    "severity": "mild/moderate/severe",
    "bounding_regions": [
        {
            "description": "fracture location",
            "relative_position": {"x": 30, "y": 40, "width": 25, "height": 20},
            "confidence": 0.85
        }
    ]
}

IMPORTANT:
- Use SIMPLE bone names only (tibia, femur, etc.)
- If no fracture, set fracture_detected to false
- Always include bounding_regions with the approximate fracture location as percentages
- Respond ONLY with valid JSON, no other text
"""
        
        if body_part_hint:
            base_prompt += f"\n\nHint: This appears to be a {body_part_hint} X-ray."
        
        return base_prompt
    
    def _parse_response(self, response_text):
        """
        Parse Gemini's response and extract structured data
        """
        try:
            # Try to extract JSON from the response
            # Sometimes the model wraps JSON in markdown code blocks
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object directly
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response_text
            
            # Parse JSON
            analysis = json.loads(json_str)
            
            # Validate and normalize the response
            return self._normalize_analysis(analysis)
            
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract information from text
            return self._extract_from_text(response_text)
    
    def _normalize_analysis(self, analysis):
        """
        Normalize and validate the analysis structure
        """
        normalized = {
            'fracture_detected': analysis.get('fracture_detected', False),
            'confidence': float(analysis.get('confidence', 0.5)),
            'body_part': analysis.get('body_part', 'unknown'),
            'fracture_type': analysis.get('fracture_type', 'not specified'),
            'location': analysis.get('location', {
                'anatomical_region': 'not specified',
                'side': 'not specified',
                'position': 'not specified'
            }),
            'severity': analysis.get('severity', 'unknown'),
            'characteristics': analysis.get('characteristics', []),
            'bounding_regions': analysis.get('bounding_regions', []),
            'recommendations': analysis.get('recommendations', 'Consult with a medical professional'),
            'notes': analysis.get('notes', '')
        }
        
        # Ensure location is a dict
        if not isinstance(normalized['location'], dict):
            normalized['location'] = {
                'anatomical_region': str(normalized['location']),
                'side': 'not specified',
                'position': 'not specified'
            }
        
        return normalized
    
    def _extract_from_text(self, text):
        """
        Fallback method to extract information from plain text response
        """
        fracture_detected = any(word in text.lower() for word in ['fracture detected', 'fracture present', 'fracture identified'])
        
        # Try to find severity
        severity = 'unknown'
        if 'severe' in text.lower():
            severity = 'severe'
        elif 'moderate' in text.lower():
            severity = 'moderate'
        elif 'mild' in text.lower():
            severity = 'mild'
        
        return {
            'fracture_detected': fracture_detected,
            'confidence': 0.7 if fracture_detected else 0.3,
            'body_part': 'detected from image',
            'fracture_type': 'detected' if fracture_detected else 'none',
            'location': {
                'anatomical_region': 'see detailed analysis',
                'side': 'not specified',
                'position': 'not specified'
            },
            'severity': severity,
            'characteristics': [text[:200]],
            'bounding_regions': [],
            'recommendations': 'Consult with a medical professional',
            'notes': text
        }
    
    def _get_fallback_response(self):
        """
        Return a fallback response when Gemini API fails
        """
        return {
            'fracture_detected': False,
            'confidence': 0.0,
            'body_part': 'unknown',
            'fracture_type': 'not analyzed',
            'location': {
                'anatomical_region': 'not analyzed',
                'side': 'not specified',
                'position': 'not specified'
            },
            'severity': 'unknown',
            'characteristics': [],
            'bounding_regions': [],
            'recommendations': 'API unavailable - using fallback detection',
            'notes': 'Gemini API analysis failed'
        }
    
    def convert_bounding_regions_to_pixels(self, bounding_regions, image_width, image_height):
        """
        Convert relative bounding box positions (percentages) to pixel coordinates
        
        Args:
            bounding_regions: List of regions with relative positions
            image_width: Width of the image in pixels
            image_height: Height of the image in pixels
            
        Returns:
            List of bounding boxes with pixel coordinates [x1, y1, x2, y2]
        """
        pixel_boxes = []
        
        for region in bounding_regions:
            try:
                rel_pos = region.get('relative_position', {})
                
                # Handle both dict and string formats
                if isinstance(rel_pos, str):
                    # Try to parse if it's a string representation
                    rel_pos = eval(rel_pos)
                
                x_percent = float(rel_pos.get('x', 30))
                y_percent = float(rel_pos.get('y', 30))
                width_percent = float(rel_pos.get('width', 40))
                height_percent = float(rel_pos.get('height', 40))
                
                # Convert to pixels
                x1 = int((x_percent / 100.0) * image_width)
                y1 = int((y_percent / 100.0) * image_height)
                x2 = int(((x_percent + width_percent) / 100.0) * image_width)
                y2 = int(((y_percent + height_percent) / 100.0) * image_height)
                
                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, image_width))
                y1 = max(0, min(y1, image_height))
                x2 = max(0, min(x2, image_width))
                y2 = max(0, min(y2, image_height))
                
                pixel_boxes.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': region.get('confidence', 0.8),
                    'description': region.get('description', 'Fracture region')
                })
                
            except Exception as e:
                print(f"Error converting bounding region: {e}")
                continue
        
        return pixel_boxes
