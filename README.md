# Arabic Sign Language Recognition System

A deep learning-based system for recognizing Arabic Sign Language (ArSL) alphabets and a selected vocabulary of words through images and camera input.

The project was developed as a Computer Science graduation project at Taibah University. It combines computer vision and deep learning models in a Flask web application to make sign recognition accessible through an interactive interface.

## Key Results

The models achieved the following accuracy on their respective test sets:

| Recognition Task | Model | Test Accuracy |
|---|---|---:|
| Alphabet Recognition (29 classes) | ArSLAttentionLSTM | **98.46%** |
| Word Recognition (10 classes) | T5-Encoder Classifier | **97.94%** |

## Project Overview

The system supports two recognition modes:

- **Alphabet Recognition:** Classifies individual Arabic sign alphabet letters from images.
- **Word Recognition:** Classifies signs from a selected vocabulary of 10 Arabic words using hand landmarks.

### Supported Word Vocabulary

| Class | Arabic Sign |
|---:|---|
| 1 | ينام |
| 2 | يسكت |
| 3 | حب |
| 4 | يدخن |
| 5 | دعم |
| 6 | مرتبك |
| 7 | قلق |
| 8 | هنا |
| 9 | السلام عليكم |
| 10 | شكراً |

## Models and Architecture

### 1. Alphabet Recognition — ArSLAttentionLSTM

A hybrid architecture combining:

- **ResNet18:** Extracts spatial image features using a pretrained backbone.
- **Bidirectional LSTM:** Models relationships between spatial feature representations.
- **Attention Mechanism:** Focuses on the most relevant feature representations.
- **Classification Head:** Predicts one of 29 alphabet classes.

### 2. Word Recognition — T5-Encoder Classifier

A landmark-based classification approach combining:

- **MediaPipe:** Extracts 21 hand landmarks, represented by x, y, and z coordinates.
- **Linear Projection:** Maps landmark features to the T5 hidden dimension.
- **T5 Encoder:** Processes the projected features.
- **Classification Head:** Predicts one of 10 word classes.

The current inference implementation extracts landmarks from a single frame and repeats the resulting feature vector to match the model's expected sequence length of 100.

## Dataset

### Alphabet Dataset

The alphabet dataset was curated from multiple public and institutional sources, followed by preprocessing and balancing.

| Dataset Split | Images |
|---|---:|
| Training | 52,084 |
| Testing | 22,359 |
| **Total** | **74,443** |

- 29 alphabet classes.
- Balanced class distribution.
- 70/30 training and testing split.

### Word Dataset

A selected vocabulary of 10 Arabic signs was used for word recognition, based on the KArSL dataset.

## Web Application Features

The Flask application provides:

- Live camera input for sign recognition.
- Static image upload and prediction.
- Video upload with frame-by-frame analysis.
- Separate alphabet and word recognition modes.
- Predicted class and confidence score.
- Inference time display.
- Video frame results with timestamps.

For uploaded videos, the application samples frames at a configured interval and processes each frame independently.

## Evaluation Results

The repository includes training and evaluation visualizations:

| File | Description |
|---|---|
| `CNN_Loss_and_Accurecy.png` | CNN training and validation curves |
| `LSTM_Loss_and_Accurecy.png` | LSTM training and validation curves |
| `LSTM_Confusion_Matrix.png` | LSTM classification performance |
| `T5_Loss_and_Accuracy.png` | T5 training and validation curves |
| `T5_Confusion_Matrix.png` | T5 classification performance |

## Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch, Torchvision |
| Computer Vision | OpenCV, MediaPipe |
| NLP / Transformers | Hugging Face Transformers, T5 |
| Web Application | Flask, HTML, CSS, JavaScript |
| Data Processing | NumPy, Pillow |

## Project Structure

```text
Arabic-Sign-Language-Translator/
│
├── model_weights/
│   ├── improved_arsl_model.pth
│   └── sign_word_t5_classifier_best_3d.pth
│
├── models/
│   ├── __init__.py
│   ├── alphabet_model.py
│   └── word_model.py
│
├── services/
│   ├── __init__.py
│   └── prediction_service.py
│
├── utils/
│   ├── __init__.py
│   ├── image_utils.py
│   └── mediapipe_utils.py
│
├── templates/
│   └── index.html
│
├── results/
│   ├── CNN_Loss_and_Accurecy.png
│   ├── LSTM_Confusion_Matrix.png
│   ├── LSTM_Loss_and_Accurecy.png
│   ├── T5_Confusion_Matrix.png
│   └── T5_Loss_and_Accuracy.png
│
├── app.py
├── config.py
├── requirements.txt
├── project_structure.md
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10 recommended.
- Git.
- A webcam (optional, for live recognition).
- GPU support is optional; the application can run on CPU.

### 1. Clone the Repository

```bash
git clone https://github.com/ash-data/Arabic-Sign-Language-Translator.git
cd Arabic-Sign-Language-Translator
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python app.py
```

Open the application in your browser:

```text
http://127.0.0.1:5000
```

## Important Notes

- Both pretrained model checkpoint files in `model_weights/` are required to run the application.
- The T5 pretrained encoder (`google-t5/t5-small`) may be downloaded automatically through Hugging Face Transformers when first loaded.
- The application selects CUDA when available; otherwise, it uses CPU.
- Camera access requires browser permission.
- The current word inference pipeline uses repeated single-frame landmarks rather than a sequence of consecutive motion frames.
- The project recognizes isolated signs and a limited vocabulary; it does not perform continuous sign language sentence translation.

## Limitations and Future Work

Potential improvements include:

- Expanding the word vocabulary.
- Supporting continuous sign language recognition.
- Training and evaluating with more diverse signers and real-world environments.
- Improving robustness to variations in lighting, hand position, and camera distance.
- Optimizing models for faster inference and resource-constrained devices.
- Conducting user acceptance testing with members of the Deaf community.

## Academic Project

**Arabic Sign Language Translator**

Graduation Project | Computer Science Department  
Taibah University | 2024–2025

The project was developed collaboratively, with contributions to research, dataset preparation, model experimentation and evaluation, application integration, testing, debugging, and technical documentation.