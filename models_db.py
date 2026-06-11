from sqlalchemy import Column, Integer, String, TIMESTAMP
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

# ---------------- USER TABLE ----------------
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False)
    password = Column(String(255), nullable=False)  # simplified for now
    created_at = Column(TIMESTAMP, server_default=func.now())


# ---------------- DETECTION HISTORY TABLE ----------------
class DetectionHistory(Base):
    __tablename__ = "detection_history"

    id = Column(Integer, primary_key=True, index=True)

    image_path = Column(String(255))

    disease_name = Column(String(255))
    confidence = Column(String(50))

    status = Column(String(50))
    severity = Column(String(50))

    recommendation = Column(String(500))  # store description
    advice = Column(String(500))          # store treatment

    created_at = Column(TIMESTAMP, server_default=func.now())