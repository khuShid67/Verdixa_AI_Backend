from fastapi import FastAPI, UploadFile, File
import shutil
import os
from database import SessionLocal
from models_db import DetectionHistory
from leaf_detector import is_leaf
from predictor import predict_leaf
from recommendations import get_recommendation
from fastapi.middleware.cors import CORSMiddleware

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI(
    title="Verdixa AI"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    db = SessionLocal()

    temp_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Step 1: Leaf check
        leaf_prediction, leaf_confidence = is_leaf(temp_path)

        if leaf_prediction == "Non_Leaf":
            return {
                "success": False,
                "status": "Not a Leaf",
                "message": "Please upload a clear plant leaf image.",
                "confidence": round(leaf_confidence, 2)
            }

        # Step 2: Disease prediction
        result = predict_leaf(temp_path)

        disease_name = result.get("prediction", "Unknown Disease")
        confidence = result.get("confidence", 0)

        # Step 3: Fix recommendation lookup (IMPORTANT FIX)
        recommendation_data = get_recommendation(disease_name)

        # Step 4: attach recommendation safely
        result["recommendation"] = recommendation_data

        status = recommendation_data.get("status", "")
        severity = recommendation_data.get("severity", "")
        description = recommendation_data.get("description", "")
        advice = recommendation_data.get("organic_treatment", "")

        result["success"] = True

        # Step 5: SAVE TO MYSQL (FIXED)
        history = DetectionHistory(
            image_path=file.filename,
            disease_name=disease_name,
            confidence=str(confidence),
            status=status,
            severity=severity,
            recommendation=description,
            advice=advice
        )

        db.add(history)
        db.commit()

        return result

    finally:
        db.close()

        if os.path.exists(temp_path):
            os.remove(temp_path)