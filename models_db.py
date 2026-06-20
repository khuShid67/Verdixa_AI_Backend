from sqlalchemy import Column, Integer, String, TIMESTAMP, Float, Text, DateTime
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

# ---------------- USER TABLE ----------------
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())


# ---------------- DETECTION HISTORY TABLE ----------------
class DetectionHistory(Base):
    __tablename__ = "detection_history"

    id = Column(Integer, primary_key=True)

    user_email = Column(String(255), nullable= False) 

    disease_name = Column(String(255), nullable=False)
    confidence = Column(Float, nullable=False)
    status = Column(String(100), nullable=False)

    image_path = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)