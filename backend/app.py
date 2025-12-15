from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
import base64
from ultralytics import YOLO
import torch
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from werkzeug.utils import secure_filename
from flask_cors import CORS  # Import CORS
import json
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from dotenv import load_dotenv
from gemini_helper import GeminiAnalyzer

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Initialize Gemini Analyzer (will use API key from .env)
try:
    gemini_analyzer = GeminiAnalyzer()
    print("✓ Gemini AI integration enabled")
    USE_GEMINI = True
except Exception as e:
    print(f"⚠ Gemini AI not available: {e}")
    print("  Falling back to traditional model only")
    gemini_analyzer = None
    USE_GEMINI = False


MODEL_PATH = os.path.join("models", "bone_fracture_model.h5")
YOLO_MODEL_PATH = os.path.join("models", "fracture_detection.pt")
BODYPART_MODEL_PATH = os.path.join("models", "bodypart_classifier.h5")

# Simplified body parts list
BODY_PARTS = {
    'hand': ['hand'],
    'arm': ['arm'],
    'leg': ['leg'],
    'foot': ['foot'],
    'chest': ['chest'],
    'spine': ['spine'],
    'skull': ['skull']
}

# Import our custom model
from models.bodypart_classifier import create_bodypart_model, preprocess_image as preprocess_bodypart_image

# Load or create models
classification_model = tf.keras.models.load_model(MODEL_PATH)

if os.path.exists(BODYPART_MODEL_PATH):
    try:
        bodypart_model = tf.keras.models.load_model(BODYPART_MODEL_PATH)
        print("Loaded existing body part classifier model")
    except Exception as e:
        print(f"Error loading existing model: {e}")
        print("Creating new body part classifier...")
        bodypart_model = create_bodypart_model(num_classes=len(BODY_PARTS))
else:
    print("Creating new body part classifier with DenseNet121...")
    bodypart_model = create_bodypart_model(num_classes=len(BODY_PARTS))

# Define the ordered class names from Kaggle model
class_names = [
    'Abdomen', 'Ankle', 'Cervical Spine', 'Chest', 'Clavicles',
    'Elbow', 'Feet', 'Finger', 'Forearm', 'Hand', 'Hip', 'Knee',
    'Lower Leg', 'Lumbar Spine', 'Others', 'Pelvis', 'Shoulder',
    'Sinus', 'Skull', 'Thigh', 'Thoracic Spine', 'Wrist'
]

# Load mapping from detailed to simplified categories
MAPPING_PATH = os.path.join('models', 'bodypart_mapping.json')
if os.path.exists(MAPPING_PATH):
    try:
        with open(MAPPING_PATH, 'r') as f:
            bodypart_mapping = json.load(f)
        print('Loaded body part mapping from JSON.')
    except Exception as e:
        print('Failed to load mapping JSON:', e)
        bodypart_mapping = {}
else:
    print('Warning: bodypart_mapping.json not found')

# Infer classifier input requirements from the loaded model
try:
    in_shape = bodypart_model.input_shape
    if isinstance(in_shape, list):
        in_shape = in_shape[0]
    _, in_h, in_w, in_c = in_shape
    classifier_input_size = (in_w, in_h)
    classifier_input_channels = in_c
    # Heuristic: Xception typically uses 299x299
    if (in_h, in_w) == (299, 299):
        classifier_preprocess = 'xception'
    else:
        classifier_preprocess = 'scale01'
    print(f'Classifier expects input {classifier_input_size} with {classifier_input_channels} channels, preprocess={classifier_preprocess}')
except Exception as e:
    print('Could not infer classifier input shape, defaulting to 224x224 grayscale:', e)
    classifier_input_size = (224, 224)
    classifier_input_channels = 1
    classifier_preprocess = 'scale01'

def prepare_classifier_input(image_gray):
    """Prepare a grayscale image (2D) for the classifier according to inferred model input.
    Returns a batched numpy array ready for model.predict
    """
    # Resize
    img = cv2.resize(image_gray, classifier_input_size)
    # Convert channels if needed
    if classifier_input_channels == 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        # keep single channel
        img = np.expand_dims(img, axis=-1)

    img = img.astype('float32')
    if classifier_preprocess == 'xception':
        # xception expects inputs in 0-255 range and applies its own scaling
        # ensure that range is 0-255
        if img.max() <= 1.0:
            img = img * 255.0
        img = xception_preprocess(img)
    else:
        # scale to [0,1]
        if img.max() > 1.0:
            img = img / 255.0

    img = np.expand_dims(img, axis=0)
    return img

# Load or create YOLO model for fracture detection
if os.path.exists(YOLO_MODEL_PATH):
    detection_model = YOLO(YOLO_MODEL_PATH)
    print(f"Loaded fracture detection model from {YOLO_MODEL_PATH}")
else:
    print("Warning: No trained fracture detection model found. Using edge detection fallback.")
    detection_model = None

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Preprocess function
def preprocess_image(image_path):
    """
    Preprocess image for both body part classification and fracture detection
    """
    # Verify file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
        
    # Read image in grayscale
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert 3D grayscale (h, w, 1) to 2D (h, w)
    if len(image_gray.shape) == 3 and image_gray.shape[2] == 1:
        image_gray = image_gray[:, :, 0]
    elif len(image_gray.shape) != 2:
        raise ValueError(f"Expected grayscale image with shape (h,w) or (h,w,1), got shape {image_gray.shape}")
    
    if image_gray.shape[0] < 32 or image_gray.shape[1] < 32:
        raise ValueError(f"Image too small: {image_gray.shape}. Minimum size is 32x32")
        
    original_size = image_gray.shape
    
    try:
        # Create color version for visualization
        image_color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        
        # Verify color conversion
        if len(image_color.shape) != 3 or image_color.shape[2] != 3:
            raise ValueError("Color conversion failed")
    except Exception as e:
        raise ValueError(f"Error converting image to color: {str(e)}")
    
    try:
        # Preprocess for both models (keeping grayscale)
        processed_image = cv2.resize(image_gray, (224, 224))
        if processed_image.shape != (224, 224):
            raise ValueError(f"Resize failed. Got shape {processed_image.shape}")
            
        # Normalize and validate range
        processed_image = processed_image / 255.0  # Normalize
        if np.min(processed_image) < 0 or np.max(processed_image) > 1:
            raise ValueError(f"Normalization failed. Values outside [0,1] range: {np.min(processed_image)} - {np.max(processed_image)}")
        
        # Add channel dimension for grayscale
        processed_image = np.expand_dims(processed_image, axis=-1)
        
        # Add batch dimension
        processed_image = np.expand_dims(processed_image, axis=0)
        
        # Final validation
        if processed_image.shape != (1, 224, 224, 1):
            raise ValueError(f"Invalid final shape: {processed_image.shape}. Expected (1, 224, 224, 1)")
            
        print("Processed image shape:", processed_image.shape)
        print("Value range:", processed_image.min(), "-", processed_image.max())
        
        return processed_image, image_color, image_gray, original_size
        
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")
    

def identify_body_part(image_gray):
    """
    Identify the body part in the X-ray image with improved confidence handling
    """
    try:
        # Prepare input according to model requirements
        model_input = prepare_classifier_input(image_gray)
        
        # Get predictions from model
        predictions = bodypart_model.predict(model_input)
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_probs = predictions[0][top_3_indices]
        
        # Get main prediction
        body_part_index = top_3_indices[0]
        confidence = float(top_3_probs[0])
        
        # Validate index is within bounds
        if not 0 <= body_part_index < len(class_names):
            raise ValueError(f"Invalid body part index: {body_part_index}")
            
        # Get the predicted body part and map to simplified category
        detailed_part = class_names[body_part_index]
        body_part = bodypart_mapping.get(detailed_part, 'unknown')
        
        # Create top 3 predictions list
        top_3_predictions = [
            {
                'part': class_names[idx],
                'simplified_category': bodypart_mapping.get(class_names[idx], 'unknown'),
                'confidence': float(predictions[0][idx]),
                'anatomy': BODY_PARTS.get(bodypart_mapping.get(class_names[idx], 'unknown'), ['unknown'])
            }
            for idx in top_3_indices
        ]
        
        print(f"Body part prediction - Part: {body_part}, Confidence: {confidence:.2f}")
        return body_part, confidence, top_3_predictions
        
    except Exception as e:
        print(f"Error in body part identification: {str(e)}")
        # Return default values if identification fails
        return "unknown", 0.0, [
            {'part': 'unknown', 'confidence': 0.0, 'anatomy': []}
        ]

def detect_fracture_location(image_color, body_part):
    """
    Detect fracture locations in the image using either YOLO model or edge detection
    Returns a list of dictionaries containing bbox coordinates and confidence
    """
    detections = []

    def _clip_box(box, W, H):
        x1, y1, x2, y2 = box
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(W - 1, int(x2))
        y2 = min(H - 1, int(y2))
        return [x1, y1, x2, y2]

    def _iou(a, b):
        # a and b are [x1,y1,x2,y2]
        xa1, ya1, xa2, ya2 = a
        xb1, yb1, xb2, yb2 = b
        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
        area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    try:
        H, W = image_color.shape[:2]
        if detection_model:
            # Use YOLO model and apply simple NMS later
            results = detection_model(image_color)
            for r in results:
                boxes = getattr(r, 'boxes', [])
                for box in boxes:
                    # ultralytics box.xyxy may be a tensor or numpy
                    coords = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else np.array(box.xyxy[0])
                    x1, y1, x2, y2 = coords
                    conf = float(getattr(box, 'conf', 0.5))
                    detections.append({'bbox': _clip_box([x1, y1, x2, y2], W, H), 'confidence': conf})
        else:
            # Fallback to edge detection for fracture detection
            gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)

            # Find contours in the edge image
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours by size to reduce noise
            min_contour_area = 80
            image_area = H * W

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_contour_area:
                    x, y, w, h = cv2.boundingRect(contour)

                    # Extract roi on edge map and find exact non-zero pixels to tighten box
                    roi = edges[y:y+h, x:x+w]
                    nz = cv2.findNonZero(roi)
                    if nz is not None and len(nz) > 0:
                        xs = nz[:, 0, 0]
                        ys = nz[:, 0, 1]
                        x_min = x + int(xs.min())
                        x_max = x + int(xs.max())
                        y_min = y + int(ys.min())
                        y_max = y + int(ys.max())

                        # Add a small padding (pixels) to the tight box
                        pad = max(3, int(0.02 * max(w, h)))
                        x1 = max(0, x_min - pad)
                        y1 = max(0, y_min - pad)
                        x2 = min(W - 1, x_max + pad)
                        y2 = min(H - 1, y_max + pad)
                    else:
                        # If no strong edges inside, fallback to bounding rect but with reduced padding
                        pad = max(3, int(0.01 * max(w, h)))
                        x1 = max(0, x + pad)
                        y1 = max(0, y + pad)
                        x2 = min(W - 1, x + w - pad)
                        y2 = min(H - 1, y + h - pad)

                    relative_size = (x2 - x1) * (y2 - y1) / float(image_area)

                    # Calculate a confidence score based on relative size and edge strength
                    roi_edges = edges[y1:y2, x1:x2]
                    edge_strength = (np.mean(roi_edges) / 255.0) if roi_edges.size > 0 else 0.0
                    confidence = float((relative_size + edge_strength) / 2.0)

                    if confidence > 0.05:  # accept weaker detections but later NMS will prune
                        detections.append({'bbox': _clip_box([x1, y1, x2, y2], W, H), 'confidence': confidence})

            # If no detections found through edge detection, try intensity-based region
            if not detections:
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    pad = max(3, int(0.02 * max(w, h)))
                    detections.append({'bbox': _clip_box([x+pad, y+pad, x+w-pad, y+h-pad], W, H), 'confidence': 0.35})
    except Exception as e:
        print(f"Warning: Error in fracture detection: {str(e)}")
        # Return at least one detection covering the suspected fracture area
        h, w = image_color.shape[:2]
        detections.append({'bbox': [w//3, h//3, 2*w//3, 2*h//3], 'confidence': 0.25})

    # Apply simple Non-Maximum Suppression to remove overlapping large boxes
    if detections:
        detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)
        keep = []
        for det in detections:
            bb = det['bbox']
            should_keep = True
            for k in keep:
                if _iou(bb, k['bbox']) > 0.3:
                    should_keep = False
                    break
            if should_keep:
                keep.append(det)
        detections = keep

    return detections

def draw_fracture_boxes(image, detections, body_part=None, image_gray=None, gemini_info=None):
    """
    Draw bounding boxes around detected fractures and highlight the body part.

    - Adds a semi-transparent fill inside each bbox to emphasize the fractured area.
    - Labels each bbox with the simplified body part name and a coarse location (3x3 grid).
    - Now enhanced with Gemini AI insights
    
    Parameters:
        image: BGR image to draw on
        detections: list of {'bbox':[x1,y1,x2,y2], 'confidence': float, 'description': str (optional)}
        body_part: simplified body part name (string)
        image_gray: grayscale image (h,w) used to compute location; if None, image is converted
        gemini_info: Optional dict with Gemini analysis results
    """
    image_with_boxes = image.copy()
    overlay = image.copy()
    H, W = image.shape[:2]

    # Ensure we have a grayscale image for location mapping
    if image_gray is None:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection.get('confidence', 0.0)
        description = detection.get('description', '')

        # Create semi-transparent overlay for the fractured area
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)

        # Compute coarse anatomical grid location based on bbox center
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        row = 0 if cy < H/3 else (1 if cy < 2*H/3 else 2)
        col = 0 if cx < W/3 else (1 if cx < 2*W/3 else 2)
        vertical_pos = ["Upper", "Middle", "Lower"][row]
        horizontal_pos = ["Left", "Center", "Right"][col]
        location_text = f"{vertical_pos} {horizontal_pos}"

        # Compose label text - use Gemini description if available
        if description:
            label = f"{description} ({confidence:.2f})"
        elif body_part and body_part != 'unknown':
            label = f"{body_part.replace('_',' ').title()} - {location_text} ({confidence:.2f})"
        else:
            label = f"Fracture - {location_text} ({confidence:.2f})"

        # Draw border and text on the resulting image
        alpha = 0.35
        cv2.addWeighted(overlay, alpha, image_with_boxes, 1 - alpha, 0, image_with_boxes)
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 0, 200), 2)

        # Put label background for readability
        (tx, ty), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(ty, 12)
        cv2.rectangle(image_with_boxes, (x1, y1 - ty - 8), (x1 + tx + 8, y1), (0, 0, 200), -1)
        cv2.putText(image_with_boxes, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add Gemini AI insights as overlay text if available
    if gemini_info and gemini_info.get('fracture_detected'):
        y_offset = 30
        insights = []
        
        if gemini_info.get('fracture_type'):
            insights.append(f"Type: {gemini_info['fracture_type']}")
        
        location_info = gemini_info.get('location', {})
        if isinstance(location_info, dict) and location_info.get('anatomical_region'):
            insights.append(f"Region: {location_info['anatomical_region']}")
        
        # Draw insights background
        for insight in insights:
            (tw, th), _ = cv2.getTextSize(insight, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image_with_boxes, (W - tw - 20, y_offset - th - 5), 
                         (W - 5, y_offset + 5), (0, 100, 0), -1)
            cv2.putText(image_with_boxes, insight, (W - tw - 15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 35

    return image_with_boxes

def get_bone_type(image, bbox):
    """
    Identify the type of bone based on the image region
    """
    x1, y1, x2, y2 = bbox
    region_height = y2 - y1
    region_width = x2 - x1
    aspect_ratio = region_width / region_height if region_height != 0 else 0

    # Analyze shape characteristics
    if aspect_ratio > 2.5:  # Long horizontal - likely ribs
        return "Rib"
    elif aspect_ratio < 0.4:  # Very vertical - likely femur/tibia/fibula
        return "Long bone"
    else:  # Other bones
        return "Other"

def get_anatomical_location(image_shape, bbox, image, body_part):
    """
    Simply returns the body part name in a readable format
    """
    # Convert body part name to readable format (replace underscores with spaces)
    readable_body_part = body_part.replace('_', ' ').title()
    
    return readable_body_part

def generate_heatmap(image, prediction_map, original_size):
    # Resize prediction map to original image size
    heatmap = cv2.resize(prediction_map, (original_size[1], original_size[0]))
    
    # Normalize heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    # Convert to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Blend original image with heatmap
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    overlayed = cv2.addWeighted(image_rgb, 0.6, heatmap, 0.4, 0)
    
    return overlayed

def analyze_severity(prediction_value, detected_area=None):
    """
    Analyze fracture severity based on both prediction confidence and fracture characteristics
    prediction_value: closer to 0 means more likely to be fractured
    """
    if prediction_value >= 0.5:  # Not fractured
        return "No fracture detected"
    
    # Convert prediction value to fracture probability (invert the scale)
    fracture_prob = 1 - prediction_value
    
    if detected_area and detected_area > 0:
        # If we have area information, use both confidence and area
        if fracture_prob > 0.85 or detected_area > 0.15:
            return "Severe"
        elif fracture_prob > 0.7 or detected_area > 0.1:
            return "Moderate"
        else:
            return "Mild"
    else:
        # If we only have confidence information
        if fracture_prob > 0.85:
            return "Severe"
        elif fracture_prob > 0.7:
            return "Moderate"
        else:
            return "Mild"

def get_fracture_location(image):
    # Divide the image into a 3x3 grid and find the region with highest intensity
    height, width = image.shape
    h_third = height // 3
    w_third = width // 3
    
    # Get average intensities for each region
    regions = []
    for i in range(3):
        for j in range(3):
            region = image[i*h_third:(i+1)*h_third, j*w_third:(j+1)*w_third]
            regions.append((np.mean(region), i, j))
    
    # Find the region with highest intensity
    _, row, col = max(regions)
    
    # Map to anatomical positions
    vertical_pos = ["Upper", "Middle", "Lower"][row]
    horizontal_pos = ["Left", "Center", "Right"][col]
    
    return f"{vertical_pos} {horizontal_pos}"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save uploaded file to the server
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Read and encode the original image
        with open(file_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Preprocess the image and get original image data
        processed_image, image_color, image_gray, original_size = preprocess_image(file_path)
        
        # Initialize Gemini analysis result
        gemini_analysis = None
        
        try:
            # Identify body part using the grayscale image (existing model)
            body_part, body_part_confidence, top_3_predictions = identify_body_part(image_gray)
            
            # Get fracture classification prediction (existing model)
            prediction = classification_model.predict(processed_image)
            prediction_value = float(prediction[0][0])
            
            # Print debug information
            print(f"Traditional Model - Raw prediction: {prediction_value}")
            print(f"Traditional Model - Body part: {body_part} ({body_part_confidence:.2f})")
            
            # Run Gemini AI analysis if available
            if USE_GEMINI and gemini_analyzer:
                print("Running Gemini AI analysis...")
                try:
                    gemini_analysis = gemini_analyzer.analyze_xray(file_path, body_part_hint=body_part)
                    print(f"Gemini AI - Fracture detected: {gemini_analysis.get('fracture_detected')}")
                    print(f"Gemini AI - Confidence: {gemini_analysis.get('confidence'):.2f}")
                    print(f"Gemini AI - Location: {gemini_analysis.get('location')}")
                    
                    # Use Gemini's analysis as primary if confidence is high
                    if gemini_analysis.get('confidence', 0) > 0.6:
                        fracture_detected = gemini_analysis.get('fracture_detected', False)
                        # Update body part if Gemini has higher confidence
                        if gemini_analysis.get('body_part') and gemini_analysis.get('body_part') != 'unknown':
                            body_part_gemini = gemini_analysis.get('body_part')
                            # Keep the traditional model's body part but note Gemini's finding
                            print(f"Gemini identified body part: {body_part_gemini}")
                    else:
                        # Fall back to traditional model
                        fracture_detected = prediction_value < 0.5
                except Exception as e:
                    print(f"Gemini analysis failed: {e}")
                    gemini_analysis = None
                    fracture_detected = prediction_value < 0.5
            else:
                # Use traditional model prediction
                fracture_detected = prediction_value < 0.5
            
            print(f"Final decision - Fracture detected: {fracture_detected}")
            
            # Add annotations to image
            # Add body part text
            cv2.putText(image_color, 
                       f"Body Part: {body_part.replace('_', ' ').title()}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            cv2.putText(image_color, f"Confidence: {body_part_confidence:.2f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Initialize variables
            detections = []
            location = "No fracture detected"
            severity = "No fracture detected"
            fracture_type = "None"
            
        except Exception as e:
            print(f"Error in detection: {str(e)}")
            body_part = "unknown"
            body_part_confidence = 0.0
            top_3_predictions = []
            detections = []
            location = "Error in detection"
            severity = "Unknown"
            fracture_detected = False
            gemini_analysis = None
        
        if fracture_detected:
            # Prioritize Gemini's bounding regions if available
            if gemini_analysis and gemini_analysis.get('bounding_regions'):
                print("Using Gemini AI bounding regions")
                H, W = image_color.shape[:2]
                detections = gemini_analyzer.convert_bounding_regions_to_pixels(
                    gemini_analysis['bounding_regions'], W, H
                )
                
                # Get simple bone name from Gemini
                body_part_from_gemini = gemini_analysis.get('body_part', '')
                if body_part_from_gemini and body_part_from_gemini != 'unknown':
                    location = body_part_from_gemini.capitalize()
                else:
                    location = body_part.replace('_', ' ').title()
                
                # Get severity and type from Gemini
                severity = gemini_analysis.get('severity', 'Unknown').title()
                fracture_type = gemini_analysis.get('fracture_type', 'Not specified')
                
            else:
                # Fall back to traditional detection
                print("Using traditional fracture detection")
                detections = detect_fracture_location(image_color, body_part)
            
            if detections:
                # If using traditional detection, get location
                if not gemini_analysis or not gemini_analysis.get('bounding_regions'):
                    main_detection = max(detections, key=lambda x: x['confidence'])
                    location = get_anatomical_location(original_size, main_detection['bbox'], image_gray, body_part)
                    
                    # Determine severity based on detection confidence and size
                    bbox = main_detection['bbox']
                    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    image_area = original_size[0] * original_size[1]
                    relative_size = bbox_area / image_area
                    
                    if relative_size > 0.2 or main_detection['confidence'] > 0.8:
                        severity = "Severe"
                    elif relative_size > 0.1 or main_detection['confidence'] > 0.6:
                        severity = "Moderate"
                    else:
                        severity = "Mild"
                
                # Create visualization with bounding boxes and labels
                marked_image = draw_fracture_boxes(image_color, detections, body_part, image_gray, gemini_analysis)
                
                # Add location text to the image
                cv2.putText(marked_image, f"Location: {location}", 
                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, (0, 0, 255), 2)
            else:
                marked_image = image_color
                if gemini_analysis:
                    severity = gemini_analysis.get('severity', 'Unknown').title()
        else:
            marked_image = image_color
        
        # Encode the marked image
        _, marked_image_encoded = cv2.imencode('.png', marked_image)
        marked_image_base64 = base64.b64encode(marked_image_encoded).decode('utf-8')

        # Prepare detection results for frontend
        detection_results = [{
            'bbox': d['bbox'],
            'confidence': d['confidence'],
            'description': d.get('description', '')
        } for d in detections]
        
        # Build enhanced response with Gemini insights
        response_data = {
            'filename': filename,
            'fracture_detected': bool(fracture_detected),
            'severity': severity,
            'location': location,
            'bodyPart': {
                'name': body_part.replace('_', ' ').title(),
                'confidence': float(body_part_confidence),
                'alternativePredictions': top_3_predictions
            },
            'originalImage': encoded_image,
            'heatmap': marked_image_base64,
            'prediction_probability': float(prediction_value),
            'detections': detection_results
        }
        
        # Add Gemini-specific insights if available
        if gemini_analysis:
            response_data['ai_analysis'] = {
                'fracture_type': gemini_analysis.get('fracture_type', 'Not specified'),
                'characteristics': gemini_analysis.get('characteristics', []),
                'recommendations': gemini_analysis.get('recommendations', ''),
                'confidence': gemini_analysis.get('confidence', 0.0),
                'enhanced_by_ai': True
            }
        else:
            response_data['ai_analysis'] = {
                'enhanced_by_ai': False
            }

        return jsonify(response_data)
        
    except Exception as e:
        print("Error processing image:", str(e))
        print("Full error details:", e.__class__.__name__)
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'error': 'Error processing image',
            'errorType': e.__class__.__name__,
            'details': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
