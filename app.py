from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import shutil
import os
import uuid

from database import SessionLocal
from models_db import DetectionHistory, User
from leaf_detector import is_leaf
from predictor import predict_leaf
from recommendations import get_recommendation


# ---------------- APP ----------------
app = FastAPI(title="Verdixa AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


# ---------------- USER ----------------
class UserRegister(BaseModel):
    email: str
    password: str


# ---------------- REGISTER ----------------
@app.post("/register")
def register(user: UserRegister):
    db = SessionLocal()
    try:
        if db.query(User).filter(User.email == user.email).first():
            return {"success": False, "message": "User exists"}

        db.add(User(email=user.email, password=user.password))
        db.commit()

        return {"success": True, "message": "Registered"}

    finally:
        db.close()


# ---------------- LOGIN ----------------
@app.post("/login")
def login(user: UserRegister):
    db = SessionLocal()
    try:
        u = db.query(User).filter(
            User.email == user.email,
            User.password == user.password
        ).first()

        if not u:
            return {"success": False, "message": "Invalid"}

        return {"success": True, "email": u.email}

    finally:
        db.close()


# ---------------- PREDICT ----------------
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    user_email: str = Form(None)
):

    db = SessionLocal()
    temp_path = None

    try:
        is_logged_in = user_email is not None

        # ---------------- GUEST LIMIT ----------------
        if not is_logged_in:
            scan_count = db.query(DetectionHistory).filter(
                DetectionHistory.user_email == "guest"
            ).count()

            if scan_count >= 3:
                return {
                    "success": False,
                    "message": "Free limit reached",
                    "used_scans": scan_count
                }

        # ---------------- SAVE IMAGE ----------------
        filename = f"{uuid.uuid4()}_{file.filename}"
        temp_path = os.path.join(UPLOAD_DIR, filename)

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # ---------------- MODEL ----------------
        leaf_prediction, leaf_confidence = is_leaf(temp_path)

        if leaf_prediction == "Non_Leaf":
            return {
                "success": False,
                "message": "Not a leaf",
                "confidence": round(leaf_confidence, 2)
            }

        result = predict_leaf(temp_path)

        disease_name = result.get("prediction", "Unknown")
        confidence = result.get("confidence", 0)

        recommendation = get_recommendation(disease_name)

        result["recommendation"] = recommendation
        result["success"] = True

        # ---------------- SAVE ONLY BASIC DATA ----------------
        if is_logged_in:
            db.add(DetectionHistory(
                user_email=user_email,
                disease_name=disease_name,
                confidence=confidence,
                image_path=filename   # IMPORTANT: stored without deleting file
            ))
            db.commit()

        return result

    finally:
        db.close()
        # ❌ REMOVED: file deletion (was causing empty uploads folder & 404)
        # Keeping file is required for Flutter history + image loading


# ---------------- HISTORY ----------------
@app.get("/history")
def get_history(user_email: str):
    db = SessionLocal()
    try:
        records = db.query(DetectionHistory).filter(
            DetectionHistory.user_email == user_email
        ).order_by(DetectionHistory.id.desc()).all()

        return [
            {
                "image_path": f"/uploads/{r.image_path}",
                "disease_name": r.disease_name,
                "confidence": r.confidence,
                "created_at": r.created_at.isoformat() if r.created_at else None
            }
            for r in records
        ]

    finally:
        db.close()


# ---------------- RECOMMENDATION ----------------
@app.get("/recommendation")
def recommendation(disease: str):
    return get_recommendation(disease)