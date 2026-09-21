from ultralytics import YOLO
import os

def train_fingerspelling():
    print("Initializing YOLO-Nano for BdSL Fingerspelling...")
    
    # Load the base classification model
    model = YOLO('yolov8n-cls.pt')

    # Train the model on your downloaded dataset
    results = model.train(
        data=os.path.abspath('data/fingerspell/'),
        epochs=50,
        imgsz=128,
        batch=64,
        lr0=0.001,
        augment=True,
        project='models',
        name='yolo_fingerspell_run'
    )
    
    print("Training complete! Your weights are saved in models/yolo_fingerspell_run/weights/best.pt")

if __name__ == '__main__':
    train_fingerspelling()