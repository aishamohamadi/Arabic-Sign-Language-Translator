# services/prediction_service.py
import torch
import numpy as np
import logging
from PIL import Image

# Import from project structure
import config
from models.alphabet_model import ArSLAttentionLSTM
from models.word_model import T5EncoderClassifier, get_t5_hidden_size
from utils import mediapipe_utils, image_utils

# --- Global Model Holders ---
_alphabet_model = None
_word_model = None

# --- Model Loading ---
def load_models():
    """Loads both alphabet and word models."""
    global _alphabet_model, _word_model
    if _alphabet_model is None:
        _alphabet_model = _load_alphabet_model_internal()
    if _word_model is None:
        _word_model = _load_word_model_internal()

def _load_alphabet_model_internal():
    """Loads the Alphabet CNN-LSTM model."""
    logging.info(f"Loading Alphabet model from: {config.ALPHABET_MODEL_PATH}")
    if not config.ALPHABET_MODEL_PATH.exists():
        logging.error(f"Alphabet model file not found at {config.ALPHABET_MODEL_PATH}")
        return None
    try:
        model = ArSLAttentionLSTM(
            num_classes=config.ALPHABET_NUM_CLASSES,
            **config.ALPHABET_MODEL_CONFIG
        ).to(config.DEVICE)

        # Load state dict - handle different checkpoint formats
        checkpoint = torch.load(config.ALPHABET_MODEL_PATH, map_location=config.DEVICE)
        state_dict = checkpoint
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            logging.info("Loading state_dict from 'model_state_dict' key.")
            # Optional: Check class count consistency if available in checkpoint
            if 'class_names' in checkpoint:
                 loaded_count = len(checkpoint['class_names'])
                 if loaded_count != config.ALPHABET_NUM_CLASSES:
                     logging.warning(f"Alphabet model checkpoint class count ({loaded_count}) differs from config ({config.ALPHABET_NUM_CLASSES}). Using config.")
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint: # Another common key
             state_dict = checkpoint['state_dict']
             logging.info("Loading state_dict from 'state_dict' key.")

        # Adjust keys if necessary (e.g., remove 'module.' prefix if saved with DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict, strict=False) # Use strict=False initially if unsure

        model.eval()
        logging.info("Alphabet model loaded successfully and set to evaluation mode.")
        return model
    except Exception as e:
        logging.error(f"Error loading Alphabet model: {e}", exc_info=True)
        return None

def _load_word_model_internal():
    """Loads the Word T5-Encoder model."""
    logging.info(f"Loading Word model from: {config.WORD_MODEL_PATH}")
    if not config.WORD_MODEL_PATH.exists():
        logging.error(f"Word model file not found at {config.WORD_MODEL_PATH}")
        return None
    try:
        # Determine projected dim dynamically if possible
        projected_dim = get_t5_hidden_size(config.WORD_BASE_MODEL_HF, config.WORD_PROJECTED_DIM_FALLBACK)

        model = T5EncoderClassifier(
            feature_dim=config.WORD_FEATURE_DIM,
            projected_dim=projected_dim, # Use determined dim
            model_name_hf=config.WORD_BASE_MODEL_HF,
            num_classes=config.WORD_NUM_CLASSES,
            dropout_rate=config.WORD_MODEL_CONFIG["dropout_rate"]
        ).to(config.DEVICE)

        # Load state dictionary
        state_dict = torch.load(config.WORD_MODEL_PATH, map_location=config.DEVICE)
        # Adjust keys if necessary (e.g., remove 'module.' prefix)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        model.eval()
        logging.info("Word model loaded successfully and set to evaluation mode.")
        return model
    except Exception as e:
        logging.error(f"Error loading Word model: {e}", exc_info=True)
        return None

# --- Prediction Functions ---

def predict_alphabet(image_pil: Image.Image):
    """Predicts Arabic alphabet sign from a PIL image."""
    if _alphabet_model is None:
        logging.error("Alphabet model not loaded.")
        return None, None, "Model not loaded"
    if image_pil is None:
        return None, None, "Invalid image"

    image_cv2 = image_utils.pil_to_cv2(image_pil)

    # 1. Optional: Hand detection check
    if not mediapipe_utils.detect_hands(image_cv2):
        logging.debug("Alphabet prediction: No hand detected.")
        return config.ARABIC_NO_HAND, 0.0, "no_hand_detected" # Return specific label and 0 confidence

    # 2. Preprocess image
    try:
        img_tensor = config.ALPHABET_TRANSFORM(image_pil).unsqueeze(0).to(config.DEVICE)
    except Exception as e:
        logging.error(f"Error transforming alphabet image: {e}", exc_info=True)
        return None, None, "Preprocessing error"

    # 3. Run inference
    try:
        with torch.no_grad():
            output = _alphabet_model(img_tensor)
            probabilities = torch.softmax(output[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)

        pred_idx_item = predicted_idx.item()
        pred_label = config.ALPHABET_CLASS_MAPPING.get(pred_idx_item, config.ARABIC_UNKNOWN)
        conf_score = round(confidence.item() * 100, 2)

        logging.debug(f"Alphabet Prediction: {pred_label} (Index: {pred_idx_item}), Confidence: {conf_score}%")

        # Apply confidence threshold if desired
        if config.PREDICTION_THRESHOLD and conf_score < config.PREDICTION_THRESHOLD * 100:
            logging.debug(f"Prediction below threshold ({config.PREDICTION_THRESHOLD*100}%)")
            return pred_label, conf_score, "ok"

        return pred_label, conf_score, "ok" # Status 'ok'

    except Exception as e:
        logging.error(f"Error during alphabet prediction inference: {e}", exc_info=True)
        return None, None, "Inference error"

def predict_word(image_pil: Image.Image):
    """Predicts Arabic word sign from a PIL image using keypoints."""
    if _word_model is None:
        logging.error("Word model not loaded.")
        return None, None, "Model not loaded"
    if image_pil is None:
        return None, None, "Invalid image"

    image_cv2 = image_utils.pil_to_cv2(image_pil)

    # 1. Extract Keypoints
    keypoints = mediapipe_utils.extract_normalized_landmarks(
        image_cv2,
        config.WORD_NUM_HANDS,
        config.WORD_COORDS_DIM
    )

    if keypoints is None:
        logging.debug("Word prediction: Keypoint extraction failed or no hand detected.")
        return config.ARABIC_NO_HAND, 0.0, "no_hand_detected" # Return specific label and 0 confidence

    # 2. Prepare for Model (Replicate single frame across sequence length)
    try:
        # Reshape keypoints assuming shape (feature_dim,)
        keypoints = keypoints.reshape(1, -1) # Ensure it's 2D: (1, feature_dim)
        # Tile across sequence length
        repeated_keypoints = np.tile(keypoints, (config.WORD_MAX_SEQ_LENGTH, 1)) # Shape: (max_seq, feature_dim)

        # Add batch dimension and convert to tensor
        input_tensor = torch.tensor(repeated_keypoints, dtype=torch.float32).unsqueeze(0).to(config.DEVICE) # Shape: (1, max_seq, feature_dim)

        # Attention mask: Treat all replicated steps as valid (T5 expects this)
        attention_mask = torch.ones(1, config.WORD_MAX_SEQ_LENGTH, dtype=torch.long).to(config.DEVICE) # Shape: (1, max_seq)
    except Exception as e:
        logging.error(f"Error preparing word model input: {e}", exc_info=True)
        return None, None, "Input prep error"

    # 3. Run Inference
    try:
        with torch.no_grad():
            outputs = _word_model(input_features=input_tensor, attention_mask=attention_mask)
            logits = outputs.logits # Shape: (1, num_classes)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)

        pred_idx_item = predicted_idx.item()
        pred_label = config.WORD_IDX_TO_LABEL_NAME.get(pred_idx_item, config.ARABIC_UNKNOWN)
        # Use probabilities[0, pred_idx_item] for confidence corresponding to max logit
        conf_score = round(probabilities[0, pred_idx_item].item() * 100, 2) # Correct confidence calculation

        logging.debug(f"Word Prediction: {pred_label} (Index: {pred_idx_item}), Confidence: {conf_score}%")

        # Apply confidence threshold if desired
        if config.PREDICTION_THRESHOLD and conf_score < config.PREDICTION_THRESHOLD * 100:
            logging.debug(f"Prediction below threshold ({config.PREDICTION_THRESHOLD*100}%)")
            return pred_label, conf_score, "ok"

        return pred_label, conf_score, "ok" # Status 'ok'

    except Exception as e:
        logging.error(f"Error during word prediction inference: {e}", exc_info=True)
        return None, None, "Inference error"