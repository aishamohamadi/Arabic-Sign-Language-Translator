# utils/mediapipe_utils.py
import mediapipe as mp
import numpy as np
import cv2
import logging
import threading
import config # Import config

mp_hands_solution = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Global detector instance (lazy initialization)
_hands_detector = None
_detector_lock = threading.Lock()  # Add a lock for thread safety

def get_hands_detector():
    """Initializes and returns the MediaPipe Hands detector."""
    global _hands_detector
    
    # Use lock to prevent concurrent initialization
    with _detector_lock:
        if _hands_detector is None:
            logging.info("Initializing MediaPipe Hands detector...")
            _hands_detector = mp_hands_solution.Hands(
                static_image_mode=config.MP_STATIC_IMAGE_MODE,
                max_num_hands=config.MP_MAX_NUM_HANDS,
                min_detection_confidence=config.MP_MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=config.MP_MIN_TRACKING_CONFIDENCE
            )
            logging.info("MediaPipe Hands detector initialized.")
    return _hands_detector

def close_hands_detector():
    """Closes the MediaPipe Hands detector."""
    global _hands_detector
    
    # Use lock to prevent concurrent access
    with _detector_lock:
        if _hands_detector is not None:
            try:
                logging.info("Closing MediaPipe Hands detector...")
                _hands_detector.close()
            except Exception as e:
                logging.error(f"Error closing MediaPipe Hands detector: {e}")
            finally:
                _hands_detector = None
                logging.info("MediaPipe Hands detector closed.")

def detect_hands(image_np_bgr):
    """
    Checks if any hands are detected in the image.
    Returns True if hands are found, False otherwise.
    """
    if image_np_bgr is None: 
        return False
        
    try:
        # Get the global singleton detector
        hands_detector = get_hands_detector()
        
        # Process the image
        image_rgb = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False # Performance optimization
        
        # Use lock to prevent concurrent access to the detector
        with _detector_lock:
            results = hands_detector.process(image_rgb)
            
        image_rgb.flags.writeable = True

        return results.multi_hand_landmarks is not None
    except Exception as e:
        logging.error(f"Error during hand detection: {e}", exc_info=True)
        return False

def extract_normalized_landmarks(image_np_bgr, num_hands_expected, coords_dim):
    """
    Extracts normalized hand landmarks for the specified number of hands.
    Returns a flattened NumPy array of shape (num_hands_expected * 21 * coords_dim,)
    """
    if image_np_bgr is None: 
        return None
        
    required_feature_length = num_hands_expected * config.WORD_NUM_HAND_LANDMARKS * coords_dim

    try:
        # Get the global singleton detector
        hands_detector = get_hands_detector()
        
        image_rgb = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Use lock to prevent concurrent access to the detector
        with _detector_lock:
            results = hands_detector.process(image_rgb)
            
        image_rgb.flags.writeable = True

        all_hand_landmarks = []
        if results.multi_hand_landmarks:
            # Process detected hands up to the expected number
            for i in range(min(len(results.multi_hand_landmarks), num_hands_expected)):
                hand_landmarks = results.multi_hand_landmarks[i]
                landmarks_list = hand_landmarks.landmark

                # Normalize relative to wrist and scale
                coords = np.array([(lm.x, lm.y, lm.z) for lm in landmarks_list])
                wrist = coords[0]
                normalized = coords - wrist

                # Scaling (use distance between wrist and MCP of middle finger)
                middle_mcp = coords[9]
                scale_dist = np.linalg.norm(middle_mcp - wrist)
                if scale_dist < 1e-6: # Avoid division by zero
                    logging.warning("Scale distance too small, skipping normalization for this hand.")
                    # Add zeros for this hand if normalization fails
                    all_hand_landmarks.append(np.zeros(config.WORD_NUM_HAND_LANDMARKS * coords_dim, dtype=np.float32))
                    continue

                normalized /= scale_dist

                # Select coordinates based on coords_dim (x,y or x,y,z)
                if coords_dim == 2:
                    selected_coords = normalized[:, :2] # Select only x, y
                else: # coords_dim == 3
                    selected_coords = normalized # Keep x, y, z

                all_hand_landmarks.append(selected_coords.flatten())

        # Pad with zeros if fewer hands were detected than expected
        num_detected = len(all_hand_landmarks)
        while len(all_hand_landmarks) < num_hands_expected:
            logging.debug(f"Padding with zeros for missing hand {len(all_hand_landmarks) + 1}")
            all_hand_landmarks.append(np.zeros(config.WORD_NUM_HAND_LANDMARKS * coords_dim, dtype=np.float32))

        # Concatenate all landmarks (detected + padded)
        if not all_hand_landmarks:
            logging.warning("No hand landmarks extracted or padded.")
            return None

        final_landmarks = np.concatenate(all_hand_landmarks).astype(np.float32)

        # Final check for correct shape
        if final_landmarks.shape[0] != required_feature_length:
            logging.error(f"Final landmark feature length mismatch! Expected {required_feature_length}, got {final_landmarks.shape[0]}")
            return None

        return final_landmarks

    except Exception as e:
        logging.error(f"Error during landmark extraction: {e}", exc_info=True)
        return None