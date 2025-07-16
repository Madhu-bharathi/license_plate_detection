# backend/app.py

import uvicorn
from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import shutil
import os

from database import SessionLocal, engine
from models import Base, VehicleLog
from yolo_easyocr import detect_license_plate_text


# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Route to upload image and perform detection
@app.post("/upload/")
async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    contents = await file.read()
    file_path = f"temp_{file.filename}"

    with open(file_path, "wb") as f:
        f.write(contents)

    result = detect_license_plate_text(file_path)
    

    new_log = VehicleLog(
      filename=file.filename,
      license_plate=result["license_plate"],
      driver_name="Unknown",  # You can improve this later
      asset_id="Unknown",     # Optional fallback for now
)

    db.add(new_log)
    db.commit()

    # Clean up temp image
    os.remove(file_path)

    return {"message": "Detection successful", "data": result}

# Route to fetch all logs
@app.get("/logs/")
def get_logs(db: Session = Depends(get_db)):
    logs = db.query(VehicleLog).all()
    return logs

# Run with uvicorn
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
