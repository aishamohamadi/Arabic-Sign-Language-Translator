# Arabic Sign Language Recognition Project Structure

```
arabic_sign_language_app/
├── model_weights/             # Trained model files
│   ├── improved_arsl_model.pth          # Alphabet recognition model
│   └── sign_word_t5_classifier_best_3d.pth  # Word recognition model
├── static/                    # Static web assets
│   ├── css/                   # Stylesheet files
│   ├── js/                    # JavaScript files
│   │   └── main.js            # Client-side functionality
│   └── favicon.ico            # Website icon
├── templates/                 # HTML templates
│   └── index.html             # Main interface
├── models/                    # Model definitions
│   ├── __init__.py            # Package initialization
│   ├── alphabet_model.py      # Attention-LSTM model for alphabet recognition
│   └── word_model.py          # T5-Encoder model for word recognition
├── services/                  # Core business logic
│   ├── __init__.py            # Package initialization
│   └── prediction_service.py  # Model loading and prediction services
├── utils/                     # Helper utilities
│   ├── __init__.py            # Package initialization
│   ├── image_utils.py         # Image processing utilities
│   └── mediapipe_utils.py     # MediaPipe hand tracking utilities
├── app.py                     # Main Flask application
├── config.py                  # Configuration settings
└── requirements.txt           # Project dependencies

Additional files:
├── academic_paper_sections.md # Academic documentation
└── README.md                  # Project overview and setup instructions
```

## Key Components

### Models
- **alphabet_model.py**: Contains the `ArSLAttentionLSTM` class that combines ResNet18, bidirectional LSTM, and attention mechanisms for Arabic alphabet sign recognition.
- **word_model.py**: Implements the `T5EncoderClassifier` class that uses MediaPipe landmarks with a T5 transformer encoder for Arabic word sign recognition.

### Services
- **prediction_service.py**: Handles model loading, inference pipeline, and provides unified prediction interfaces for both alphabet and word recognition.

### Utilities
- **image_utils.py**: Provides functionality for image conversion, processing, and base64 encoding/decoding.
- **mediapipe_utils.py**: Implements hand detection and landmark extraction using MediaPipe.

### Web Interface
- **index.html**: Responsive user interface with camera integration and image upload capabilities.
- **main.js**: Client-side functionality for camera control and prediction display.

### Configuration
- **config.py**: Contains all configurable parameters, paths, and settings used throughout the application.

### Application
- **app.py**: Flask application that serves the web interface and prediction API endpoints.
