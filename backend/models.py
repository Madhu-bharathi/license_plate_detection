# backend/models.py

from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
from database import Base

class VehicleLog(Base):
    __tablename__ = "vehicle_logs"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)  # ✅ ADD THIS
    license_plate = Column(String, index=True)
    driver_name = Column(String)
    asset_id = Column(String)
    gate_number = Column(String, default="Unknown")  # Optional fallback
    coordinates = Column(String, default="Unknown")
    entry_time = Column(DateTime, default=datetime.utcnow)

