from ultralytics import YOLO
import easyocr
import cv2
import os
from datetime import datetime

# Load the pretrained license plate model
model = YOLO("license_plate_detector.pt")
reader = easyocr.Reader(['en'])
os.makedirs("captured", exist_ok=True)

def detect_license_plate_text(image_path):
    image = cv2.imread(image_path)
    results = model(image)[0]

    cropped_texts = []
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = image[y1:y2, x1:x2]

        snapshot = f"captured/plate_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.jpg"
        cv2.imwrite(snapshot, cropped)

        ocr = reader.readtext(cropped)
        if ocr:
            cropped_texts.append(ocr[0][1])

    return {
        "license_plate": cropped_texts[0] if cropped_texts else "Unknown"
    }
