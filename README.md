# BdSL Real-Time Translation System - V1

A desktop application designed for real-time Bangladeshi Sign Language (BdSL) translation. This project leverages deep learning, computer vision, and a dynamic stream-routing architecture to detect and translate both static fingerspelling and dynamic whole-word gestures into text.

## Features
*   **Real-Time PyQt5 UI:** A clean, responsive dashboard featuring a live webcam feed, current detection mode, and a live transcript history box.
*   **Multi-Threaded Video Capture:** Utilizes a custom threading implementation (`CaptureThread`) with OpenCV to maintain a stable 30 FPS webcam feed without blocking the main UI.
*   **MediaPipe Landmark Extraction:** Extracts a robust 258-dimensional feature vector combining pose (132 points), left hand (63 points), and right hand (63 points) landmarks.
*   **Dynamic Stream Routing:** Intelligently switches between translation models based on the subject's hand bounding box area. If the hand area is under 800 px², the system routes the cropped hand ROI to the YOLO model for fingerspelling; otherwise, it defaults to the Transformer model for whole-word recognition.
*   **YOLOv8n Fingerspelling Classification:** Custom YOLOv8 Nano pipeline trained on BdSL fingerspelling datasets with built-in data augmentation.

## Repository Structure
*   `main.py`: The core PyQt5 application and execution pipeline.
*   `src/capture/capture.py`: Multi-threaded OpenCV webcam capture[cite: 1].
*   `src/landmarks/extractor.py`: MediaPipe Holistic wrapper for landmark extraction and bounding-box calculation[cite: 2].
*   `train_yolo.py`: Training script for the YOLOv8n-cls fingerspelling model.
*   `fingerspell_dataset.yaml`: Dataset configuration for YOLO[cite: 5].
*   `requirements.txt`: Python environment dependencies.

## Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YourUsername/BdSL-Translation.git](https://github.com/YourUsername/BdSL-Translation.git)
   cd BdSL-Translation

2.  **Install dependencies:**
Ensure you have Python 3.8+ installed, then run:

 ```bash
pip install -r requirements.txt

**Required packages include:** torch, torchvision, opencv-python, mediapipe, ultralytics, and PyQt5.

3. **Download or Train Models:**

Pre-trained: Ensure your trained models are placed in the models/ directory or runs/classify/models/yolo_fingerspell_run/weights/.

Train from scratch: Run python train_yolo.py to train the YOLOv8n-cls model on your local dataset. Weights will be saved to models/yolo_fingerspell_run/weights/best.pt.

4. **Usage**
To launch the real-time translation interface, execute:

 ```Bash
python main.py
The UI will initialize the webcam, load the YOLO classification model (with a fallback to the base model if custom weights are missing), and immediately begin processing hand gestures. The top prediction will display on the screen once the model hits a 60% confidence threshold.

4. **Author**
Developed by **Md. Rayhan Islam Showrav - 0112230810**.
