# utils/image_utils.py
import base64
import io
from PIL import Image
import numpy as np
import cv2 # OpenCV for color conversion if needed

def decode_base64_image(base64_string):
    """Decodes a base64 string into a PIL Image."""
    if isinstance(base64_string, str) and base64_string.startswith('data:image'):
        base64_string = base64_string.split(',')[1]
    try:
        image_bytes = base64.b64decode(base64_string)
        image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return image_pil
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None

def pil_to_cv2(pil_image):
    """Converts a PIL Image to an OpenCV image (NumPy array, BGR)."""
    if pil_image is None:
        return None
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image

def read_image_file(file_storage):
    """Reads a Flask FileStorage object into a PIL Image."""
    try:
        image_pil = Image.open(file_storage.stream).convert('RGB')
        return image_pil
    except Exception as e:
        print(f"Error reading image file: {e}")
        return None