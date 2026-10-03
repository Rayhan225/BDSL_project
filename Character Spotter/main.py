import sys
import cv2
import os
import numpy as np
from collections import deque
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QTextEdit
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from ultralytics import YOLO

# Import your custom modules
from src.capture.capture import CaptureThread
from src.landmarks.extractor import LandmarkExtractor

class BdSLTranslationUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BdSL Real-Time Translation System - V1")
        self.setGeometry(100, 100, 1000, 500)

        # --- DATASET & MODEL PATHS ---
        self.dataset_sequences_dir = "data/sequences/"
        self.dataset_fingerspell_dir = "data/fingerspell/"
        self.transformer_model_path = "models/transformer.pt"
        self.llm_model_path = "models/qwen2.5-1.5b-instruct.Q4_K_M.gguf"
        
        # --- SAFE YOLO MODEL LOADING ---
        # Updated to your exact YOLO training path
        custom_yolo_path = os.path.join("runs", "classify", "models", "yolo_fingerspell_run", "weights", "best.pt")
        fallback_yolo_path = "yolov8n-cls.pt" 

        if os.path.exists(custom_yolo_path):
            self.yolo_model_path = custom_yolo_path
            print(f"SUCCESS: Loading custom YOLO model from {custom_yolo_path}")
        else:
            self.yolo_model_path = fallback_yolo_path
            print(f"WARNING: Custom model not found. Falling back to base model: {fallback_yolo_path}")

        # Initialize the AI Model
        self.yolo_model = YOLO(self.yolo_model_path)

        # --- UI LAYOUT ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)

        # Left Panel (Live Webcam Feed)
        self.video_label = QLabel("Webcam Feed")
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: black;")
        self.layout.addWidget(self.video_label)

        # Right Panel (Translation & Transcript History)
        self.text_panel = QVBoxLayout()
        self.translation_label = QLabel("Mode: Initializing...")
        self.translation_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.transcript_box = QTextEdit()
        self.transcript_box.setReadOnly(True)
        self.transcript_box.setPlaceholderText("Live transcript history...")
        
        self.text_panel.addWidget(self.translation_label)
        self.text_panel.addWidget(self.transcript_box)
        self.layout.addLayout(self.text_panel)

        # --- PIPELINE INITIALIZATION ---
        self.cap_thread = CaptureThread(src=0)
        self.cap_thread.start()
        self.extractor = LandmarkExtractor()
        
        # Data Buffers
        self.landmark_buffer = deque(maxlen=30)
        self.token_buffer = []

        # Application Loop (30 FPS Target)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_pipeline)
        self.timer.start(33)

    def update_pipeline(self):
        frame = self.cap_thread.get_frame()
        if frame is None:
            return

        # Convert to RGB for MediaPipe extraction
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks, hand_roi, bbox_area = self.extractor.process(rgb_frame)

        if landmarks is not None:
            self.landmark_buffer.append(landmarks)
            
            # Stream Router Logic: Route to YOLO if hand is smaller than 800 px²
            if 0 < bbox_area < 800:
                if hand_roi is not None and hand_roi.size > 0:
                    
                    # Pass the cropped hand image to YOLO silently
                    results = self.yolo_model(hand_roi, verbose=False)
                    
                    # Extract the top prediction and its confidence score
                    top1_idx = results[0].probs.top1
                    confidence = results[0].probs.top1conf.item()
                    predicted_letter = results[0].names[top1_idx]
                    
                    # Only update the UI if the AI is at least 60% confident
                    if confidence > 0.60:
                        self.translation_label.setText(f"Mode: FINGERSPELL - Letter: {predicted_letter} ({confidence*100:.1f}%)")
                    else:
                        self.translation_label.setText("Mode: FINGERSPELL (Detecting...)")
            else:
                self.translation_label.setText("Mode: WHOLE_WORD (Transformer)")
                
        # Render the frame to the UI overlay
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    def closeEvent(self, event):
        self.cap_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BdSLTranslationUI()
    window.show()
    sys.exit(app.exec_())