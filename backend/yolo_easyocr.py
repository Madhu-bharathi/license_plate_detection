from ultralytics import YOLO
import easyocr
import cv2
import os
from datetime import datetime

# Load YOLO model and EasyOCR reader
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

        # Save cropped plate image
        snapshot = f"captured/plate_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.jpg"
        cv2.imwrite(snapshot, cropped)

        # Perform OCR
        ocr = reader.readtext(cropped)
        if ocr:
            print(f"[DEBUG] OCR result for box {i}:", ocr)
            # Join all detected text in the box
            cropped_texts.append(" ".join([item[1] for item in ocr]))

    # Join all detected plates into one string
    license_plate_text = " ".join(cropped_texts) if cropped_texts else "Unknown"

    return {
        "license_plate": license_plate_text
    }
