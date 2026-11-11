# config.py
from pathlib import Path
import torch
from torchvision import transforms

# --- General Settings ---
SCRIPT_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Weights Location ---
MODEL_WEIGHTS_DIR = SCRIPT_DIR / "model_weights"

# --- Alphabet Model (CNN-LSTM) Settings ---
ALPHABET_MODEL_PATH = MODEL_WEIGHTS_DIR / "improved_arsl_model.pth"
# Ensure this mapping matches the *exact* output order of trained alphabet model
ALPHABET_CLASS_MAPPING = {
     0: 'ع',     1: 'أ',     2: 'ب',     3: 'د',     4: 'ظ',  
     5: 'ض',     6: 'ف',     7: 'ق',     8: 'غ',     9: 'ه',
    10: 'ح',    11: 'ج',    12: 'ك',    13: 'خ',    14: 'لا',
    15: 'ل',    16: 'م',    17: 'ن',    18: 'ر',    19: 'ص',
    20: 'س',    21: 'ش',    22: 'ط',    23: 'ت',    24: 'ث',
    25: 'ذ',    26: 'و',    27: 'ي',    28: 'ز'
    
}
ALPHABET_NUM_CLASSES = len(ALPHABET_CLASS_MAPPING) # Should be 29 based on above
ALPHABET_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
ALPHABET_MODEL_CONFIG = {
    "hidden_size": 512,
    "num_layers": 2,
    "bidirectional": True,
    "dropout_rate": 0.5
}

# --- Word Model (T5-Encoder) Settings ---
WORD_MODEL_PATH = MODEL_WEIGHTS_DIR / "sign_word_t5_classifier_best_3d.pth" # Verify filename
WORD_BASE_MODEL_HF = "google-t5/t5-small" # Must match training
WORD_MAX_SEQ_LENGTH = 100 # Must match training
WORD_NUM_HAND_LANDMARKS = 21 # From mediapipe
WORD_NUM_HANDS = 1 # Set to 1 or 2 based on *trained model*. Original T5 app used 1.
WORD_COORDS_DIM = 3 # x, y, z
# If model used only x,y, set WORD_COORDS_DIM = 2
WORD_FEATURE_DIM = WORD_NUM_HAND_LANDMARKS * WORD_COORDS_DIM * WORD_NUM_HANDS # e.g., 21*3*1 = 63 or 21*3*2 = 126
# This MUST match the T5 hidden size used during training
# Determined dynamically in word_model.py, but provide a fallback
WORD_PROJECTED_DIM_FALLBACK = 512 # Fallback if dynamic loading fails (e.g., for t5-small)

# Ensure this mapping matches the *exact* output order of  trained word model
WORD_IDX_TO_LABEL_NAME = {
    0: 'ينام',    # sleep
    1: 'يسكت',    # silence
    2: 'حب',     # love
    3: 'يدخن',   # smoke
    4: 'دعم',    # support
    5: 'مرتبك',  # confused
    6: 'قلق',    # worried
    7: 'هنا',    # here
    8: 'السلام عليكم', # greeting
    9: 'شكرا',    # thanks
    
}
WORD_NUM_CLASSES = len(WORD_IDX_TO_LABEL_NAME) # Should be 10 based on above
WORD_MODEL_CONFIG = {
    "dropout_rate": 0.1
}

# --- MediaPipe Settings ---
MP_STATIC_IMAGE_MODE = False # Changed to False for video processing
MP_MAX_NUM_HANDS = 2 # Detect up to two hands
MP_MIN_DETECTION_CONFIDENCE = 0.3 # Lowered from 0.5 to potentially detect more hands
MP_MIN_TRACKING_CONFIDENCE = 0.5 # Only used if static_image_mode=False

# --- Web Application Settings ---
WEBCAM_INTERVAL_MS = 200 # How often frontend sends frame (milliseconds)
PREDICTION_THRESHOLD = 0.5 # Minimum confidence to display prediction (optional)
ARABIC_NO_HAND = "لا توجد يد"
ARABIC_UNKNOWN = "غير معروف"
VIDEO_SAMPLE_RATE = 5 # Process every Nth frame of the video (e.g., 5 means process frame 0, 5, 10, ...)