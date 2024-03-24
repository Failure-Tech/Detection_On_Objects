import torch
import numpy as np
import cv2
import time
import easyocr
from flask import Flask, request, jsonify
from flask_cors import CORS
import csv

app = Flask(__name__)
CORS(app, origins="http://localhost:5173")

class ObjectDetection:
    def __init__(self):
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ocr_reader = easyocr.Reader(['en'])  # Initialize EasyOCR reader with English language
        self.detected_text = []  # Variable to store detected text
    
    def load_model(self):
        model = torch.hub.load("ultralytics/yolov5", 'yolov5s', pretrained=True)
        return model
    
    def score_frame(self, frame):
        # Takes single frame as input and score frame with yolov5
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)

        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:,:-1]
        return labels, cord
    
    def class_to_label(self, x):
        return self.classes[int(x)]
    
    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)  # Put text label on the bounding box
        return frame
    
    def ocr(self, frame):
        # Perform OCR on the provided frame
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(grayscale, 150, 255, cv2.THRESH_BINARY)
        text = self.ocr_reader.readtext(binary)
        detected_text = []
        for detection in text:
            bbox = detection[0]
            text = detection[1]
            x1, y1 = int(bbox[0][0]), int(bbox[0][1])
            x2, y2 = int(bbox[2][0]), int(bbox[2][1])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Put text label
            detected_text.append(text)
        return frame, detected_text
    
    def process_frame(self, frame):
        results = self.score_frame(frame)
        frame = self.plot_boxes(results, frame)
        frame, text = self.ocr(frame)  # Perform OCR on the frame
        self.detected_text.extend(text)  # Add detected text to the list
        return frame
    
    def store_text(self, detected_text):
        with open('example_store.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Detected Text'])  # Write header
            writer.writerow([detected_text])  # Write detected text with spaces
            return detected_text

# Instantiate the ObjectDetection class
detection = ObjectDetection()

# Route to start object detection
@app.route('/start_detection', methods=['POST'])
def start_detection():
    cam = cv2.VideoCapture(0)
    detected_texts = []  # List to store all detected text
    while cam.isOpened():
        start_time = time.perf_counter()
        ret, frame = cam.read()
        if not ret:
            break
        frame = detection.process_frame(frame)
        end_time = time.perf_counter()
        fps = 1/np.round(end_time - start_time, 3)
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.imshow("img", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    # Release the camera and close OpenCV windows
    cam.release()
    cv2.destroyAllWindows()

    # Capture the detected texts
    detected_texts = detection.detected_text
    detection.detected_text = []  # Clear the detected text for the next run
    return jsonify({"Message": "Object Detection run Successfully", "texts": detected_texts})

if __name__ == "__main__":
    app.run(debug=True)
