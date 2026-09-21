import cv2
import threading
import queue
import time

class CaptureThread(threading.Thread):
    def __init__(self, src=0):
        super().__init__()
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.q = queue.Queue(maxsize=5)
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            # Flip the frame horizontally (mirror mode) for a natural user experience
            frame = cv2.flip(frame, 1)
            
            if not self.q.full():
                self.q.put(frame)
            else:
                try:
                    self.q.get_nowait()
                    self.q.put(frame)
                except queue.Empty:
                    pass

    def get_frame(self):
        if not self.q.empty():
            return self.q.get()
        return None

    def stop(self):
        self.running = False
        self.join()
        self.cap.release()