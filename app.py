import os
import time
import logging
import torch
import base64
import io
import tempfile # <-- Add this
import cv2      # <-- Add this
import numpy as np # <-- Add this
from flask import Flask, request, jsonify, render_template, Response
from PIL import Image
from services import prediction_service
from utils import image_utils, mediapipe_utils
import config

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Global initialization
# Global initialization 
@app.before_first_request 
def initialize(): 
    """Initialize services on first request.""" 
    prediction_service.load_models()
# Routes
@app.route('/')
def index():
    """Renders the main page with webcam and prediction UI."""
    return render_template('index.html', capture_interval=config.WEBCAM_INTERVAL_MS)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles prediction from both webcam frames and uploaded images.
    """
    try:
        start_time = time.time()
        mode = request.form.get('mode', 'alphabet')  # Default to alphabet mode
        
        # Process image from either JSON or form data
        if request.content_type and 'application/json' in request.content_type:
            # Case 1: JSON with base64 image from webcam
            data = request.get_json()
            image_data = data.get('image')
            if not image_data:
                return jsonify({'status': 'error', 'error': 'No image data provided'}), 400
                
            mode = data.get('mode', 'alphabet')  # Get mode from JSON if provided
            image_pil = image_utils.decode_base64_image(image_data)
        else:
            # Case 2: Form data with uploaded file
            if 'image' not in request.files:
                return jsonify({'status': 'error', 'error': 'No image file provided'}), 400
                
            file = request.files['image']
            if file.filename == '':
                return jsonify({'status': 'error', 'error': 'No file selected'}), 400
                
            image_pil = image_utils.read_image_file(file)
        
        # Check if image was properly processed
        if image_pil is None:
            return jsonify({'status': 'error', 'error': 'Could not process image'}), 400
        
        # Perform prediction based on mode
        prediction = confidence = status = None
        try:
            if mode == 'alphabet':
                prediction, confidence, status = prediction_service.predict_alphabet(image_pil)
            elif mode == 'word':
                prediction, confidence, status = prediction_service.predict_word(image_pil)
            else:
                return jsonify({'status': 'error', 'error': 'Invalid mode specified'}), 400
                
            # Calculate inference time
            inference_time_ms = int((time.time() - start_time) * 1000)
            
            # Return results
            return jsonify({
                'status': status,
                'class': prediction,
                'confidence': confidence,
                'inference_time_ms': inference_time_ms
            })
        except Exception as e:
            logging.error(f"Error during prediction: {e}", exc_info=True)
            # Corrected line: Removed the extra ')' inside the f-string
            return jsonify({'status': 'error', 'error': f'Prediction error: {str(e)}'}), 500
    
    except Exception as e:
        logging.error(f"Unhandled error during request processing: {e}", exc_info=True)
        return jsonify({'status': 'error', 'error': 'Internal server error'}), 500

# Add the new route for video prediction here
@app.route('/predict_video', methods=['POST'])
def predict_video():
    """Handles prediction for uploaded video files."""
    # Use the correct variable names with leading underscores
    if not prediction_service._alphabet_model or not prediction_service._word_model:
        return jsonify({'error': 'Models not loaded or failed to load'}), 500

    video_path = None # Initialize video_path
    cap = None        # Initialize cap
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video provided'}), 400

        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Use a temporary file to save the video
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        video_path = temp_file.name
        video_file.save(video_path)
        logging.info(f"Video saved temporarily to {video_path}")

        # Extract mode parameter if provided
        mode = request.form.get('mode', 'alphabet')
        logging.info(f"Processing video in '{mode}' mode.")

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error("Could not open video file.")
            return jsonify({'error': 'Could not open video file'}), 400

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logging.info(f"Video properties: FPS={fps}, Frame Count={frame_count}")

        # Process video frames (sample every Nth frame)
        sample_rate = config.VIDEO_SAMPLE_RATE # Use config value
        results = []
        frame_num = 0
        processed_frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break # End of video

            # Only process every Nth frame
            if frame_num % sample_rate == 0:
                processed_frame_count += 1
                try:
                    # Convert frame to PIL image
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_pil = Image.fromarray(frame_rgb)
                    
                    # Create a thumbnail before any processing
                    thumb = image_pil.copy()
                    thumb.thumbnail((160, 90)) # 16:9 ratio thumbnail
                    buffered = io.BytesIO()
                    thumb.save(buffered, format="JPEG")
                    thumb_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    # Check for hand detection first using the OpenCV frame
                    # This is a critical step that was missing
                    hands_detected = mediapipe_utils.detect_hands(frame)
                    
                    if not hands_detected:
                        # No hands detected - set to Arabic "No Hand Detected" message
                        results.append({
                            'frame': frame_num,
                            'time': frame_num / fps if fps > 0 else 0,
                            'class': config.ARABIC_NO_HAND,  # Use config constant for consistency
                            'confidence': None,
                            'status': 'no_hand',
                            'thumbnail': f"data:image/jpeg;base64,{thumb_base64}"
                        })
                        # Skip to next frame without attempting prediction
                        frame_num += 1
                        continue

                    # Perform prediction based on mode only if hands were detected
                    prediction = confidence = status = None
                    if mode == 'alphabet':
                        prediction, confidence, status = prediction_service.predict_alphabet(image_pil)
                    elif mode == 'word':
                        prediction, confidence, status = prediction_service.predict_word(image_pil)
                    else:
                        # Skip frame if mode is invalid, but log it
                        logging.warning(f"Invalid mode '{mode}' specified for frame {frame_num}. Skipping.")
                        continue # Go to next frame

                    # Add result for this frame
                    results.append({
                        'frame': frame_num,
                        'time': frame_num / fps if fps > 0 else 0,
                        'class': prediction,
                        'confidence': confidence,
                        'status': status,
                        'thumbnail': f"data:image/jpeg;base64,{thumb_base64}"
                    })
                except Exception as frame_e:
                    logging.error(f"Error processing frame {frame_num}: {frame_e}", exc_info=True)
                    # Add an error entry to results
                    results.append({
                        'frame': frame_num,
                        'time': frame_num / fps if fps > 0 else 0,
                        'class': config.ARABIC_UNKNOWN,  # Use "Unknown" text in Arabic
                        'confidence': None,
                        'status': 'error',
                        'thumbnail': f"data:image/jpeg;base64,{thumb_base64}" if 'thumb_base64' in locals() else None
                    })

            frame_num += 1

        logging.info(f"Finished processing video. Processed {processed_frame_count} frames.")

        # Analyze results to find most frequent high-confidence prediction
        confidence_threshold = 70  # Only consider predictions with confidence > 70%
        prediction_counts = {}
        
        # Count occurrences of each prediction with confidence above threshold
        for result in results:
            if result['status'] == 'success' and result['confidence'] is not None and result['confidence'] > confidence_threshold:
                prediction = result['class']
                if prediction not in prediction_counts:
                    prediction_counts[prediction] = {
                        'count': 0,
                        'total_confidence': 0,
                        'highest_confidence': 0
                    }
                
                prediction_counts[prediction]['count'] += 1
                prediction_counts[prediction]['total_confidence'] += result['confidence']
                
                if result['confidence'] > prediction_counts[prediction]['highest_confidence']:
                    prediction_counts[prediction]['highest_confidence'] = result['confidence']
        
        # Find the most frequent prediction
        best_prediction = None
        highest_count = 0
        
        for pred, stats in prediction_counts.items():
            if stats['count'] > highest_count:
                highest_count = stats['count']
                best_prediction = {
                    'prediction': pred,
                    'count': stats['count'],
                    'occurrence_percent': round((stats['count'] / processed_frame_count) * 100, 1),
                    'avg_confidence': round(stats['total_confidence'] / stats['count'], 1),
                    'highest_confidence': round(stats['highest_confidence'], 1)
                }
        
        # If no predictions met the threshold, set best_prediction to None
        if not best_prediction and len(results) > 0:
            # Check if we had any hand detections at all
            hand_detected_count = sum(1 for r in results if r['status'] != 'no_hand')
            if hand_detected_count == 0:
                best_prediction = {
                    'prediction': config.ARABIC_NO_HAND,
                    'count': processed_frame_count,
                    'occurrence_percent': 100,
                    'avg_confidence': None,
                    'highest_confidence': None
                }

        return jsonify({
            'total_frames': frame_count,
            'processed_frames': processed_frame_count,
            'fps': fps,
            'results': results,
            'best_prediction': best_prediction
        })

    except Exception as e:
        logging.error(f"Video prediction error: {e}", exc_info=True)
        # The finally block will handle cleanup
        return jsonify({'error': f'An internal error occurred during video processing: {str(e)}'}), 500
    finally:
        # Ensure resources are released and temp file is deleted
        if cap is not None:
            cap.release()
            logging.info("Video capture released.")
        if video_path is not None and os.path.exists(video_path):
            try:
                os.unlink(video_path)
                logging.info(f"Temporary video file {video_path} deleted.")
            except PermissionError as unlink_e:
                 # Log the error but don't crash the request
                 logging.error(f"Error removing temp file {video_path} (might be locked): {unlink_e}")
            except Exception as unlink_e:
                 logging.error(f"Unexpected error removing temp file {video_path}: {unlink_e}")

# Add better signal handling for shutdown
import signal
import atexit

def shutdown_handler(signal=None, frame=None):
    """Clean up resources on shutdown"""
    logging.info("Shutting down application...")
    
    # Close MediaPipe detector at application shutdown
    from utils import mediapipe_utils
    mediapipe_utils.close_hands_detector()
    
    logging.info("Application shutdown complete")

# Register the shutdown handler
atexit.register(shutdown_handler)
signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

# REMOVE or comment out the teardown_appcontext handler
# since it's closing the detector too frequently
# @app.teardown_appcontext
# def cleanup(exception=None):
#     """Closes resources when the app context ends."""
#     mediapipe_utils.close_hands_detector()

if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('model_weights', exist_ok=True)
    
    # Check if model files exist
    if not config.ALPHABET_MODEL_PATH.exists():
        logging.warning(f"Alphabet model not found at {config.ALPHABET_MODEL_PATH}")
    
    if not config.WORD_MODEL_PATH.exists():
        logging.warning(f"Word model not found at {config.WORD_MODEL_PATH}")
    
    # Start server
    logging.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=False)