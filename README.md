# Onubad: BdSL Real-Time Translation System

A comprehensive AI-based Bangla Sign Language (BdSL) recognition and translation system. This project leverages computer vision, deep learning dual-stream recognition, and a quantized Large Language Model (LLM) pipeline to translate both static fingerspelled characters and dynamic whole-word gestures into text and natural speech.

---

## 🏗️ Repository Architecture

The repository is organized into three main modules following our dual-stream vision recognition and semantic fusion architecture:

```
BDSL_project/
├── Character Spotter/            # YOLOv8 Nano fingerspelling character spotter
│   ├── data/                     # Dataset directories
│   ├── models/                   # Trained model weights
│   ├── runs/                     # YOLO training runs & evaluation metrics
│   ├── src/                      # Source code (webcam capture, landmark extraction)
│   ├── fingerspell_dataset.yaml  # Dataset configuration
│   ├── main.py                   # PyQt5 real-time application entry point
│   ├── train_yolo.py             # YOLOv8n-cls training script
│   └── requirements.txt          # Module dependencies
│
├── Word Spotter/                 # Transformer-based word spotter model
│   ├── bdslp-transformer.ipynb   # Model training & evaluation notebook
│   ├── video_classifier.weights.h5 # Trained Transformer weights
│   └── requirements.txt          # Module dependencies
│
├── Quantized LLM/                # Semantic reconstruction engine (In Development)
│   └── README.md                 # Quantized edge LLM setup & prompt integration
│
├── 1.intro.tex                   # Project Introduction & Methodology LaTeX chapter
├── 4.implementation.tex          # Implementation & Results LaTeX chapter
└── README.md                     # Project documentation
```

---

## 🌟 Key Components & Models

### 1. 🔤 Character Spotter (YOLOv8 Nano)
- **Architecture:** `yolov8n-cls` classification backbone optimized for edge devices.
- **Task:** Tracking and recognizing individual fingerspelled Bangla characters.
- **Performance:**
  - **Top-1 Accuracy:** `94.21%`
  - **Top-5 Accuracy:** `99.51%`
  - **Min Validation Loss:** `0.1929`

### 2. 🔠 Word Spotter (Transformer Encoder)
- **Architecture:** Spatial-Temporal Transformer model with custom `PositionalEmbedding` and `TransformerEncoder` layers (1 attention head, key dim = 258, dropout = 0.3).
- **Features:** 30-frame sequence input using 258 MediaPipe Holistic skeletal landmark coordinates per frame $(132 \text{ pose} + 63 \text{ left hand} + 63 \text{ right hand})$.
- **Vocabulary:** 102 BdSL word classes.
- **Performance:**
  - **Overall Test Accuracy:** `95.18%`
  - **Macro Precision:** `95.34%`
  - **Macro Recall:** `95.50%`
  - **Macro F1-Score:** `95.00%`

### 3. 🧠 Quantized LLM Semantic Reconstruction *(Upcoming)*
- Local 4-bit quantized edge LLM integration (e.g., Gemma 2B / Qwen 2.5) for converting predicted character/word streams into grammatically correct Bangla sentences.

---

## 🚀 Getting Started

### Installation & Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUsername/BDSL_project.git
   cd BDSL_project
   ```

2. **Set up Python Environment:**
   Ensure you have Python 3.10 installed:
   ```bash
   pip install -r "Character Spotter/requirements.txt"
   ```

3. **Running the Character Spotter Application:**
   ```bash
   cd "Character Spotter"
   python main.py
   ```

4. **Training / Evaluating Word Spotter:**
   Open and execute `Word Spotter/bdslp-transformer.ipynb` in Jupyter Notebook or VS Code.

---

## 📄 Research & Documentation
The complete methodology, system design, hardware setup, and empirical evaluation matrices are documented in LaTeX:
- `1.intro.tex`: Project overview, dual-stream architecture, and motivation.
- `4.implementation.tex`: Comprehensive experimental setup, hardware specs, and accuracy evaluation tables.

---

Developed by **Team Auditory Cortex**.