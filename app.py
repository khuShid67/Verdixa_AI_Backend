from fastapi import FastAPI, UploadFile, File
import shutil
import os

from leaf_detector import is_leaf
from predictor import predict_leaf
from recommendations import get_recommendation
from fastapi.middleware.cors import CORSMiddleware


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

    temp_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    try:
        # Save uploaded image
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # -----------------------------
        # Leaf Detection
        # -----------------------------
        leaf_prediction, leaf_confidence = is_leaf(temp_path)

        print(
            f"Leaf Detector: {leaf_prediction} "
            f"({leaf_confidence:.2f}%)"
        )

        if leaf_prediction == "Non_Leaf":

            return {
                "success": False,
                "status": "Not a Leaf",
                "message": "Please upload a clear image of a plant leaf.",
                "confidence": round(leaf_confidence, 2)
            }

        # -----------------------------
        # Disease Detection
        # -----------------------------
        result = predict_leaf(temp_path)

        print("RESULT:", result)

        # -----------------------------
        # Recommendations
        # -----------------------------
        if result["prediction"] != "Unknown Disease":

            recommendation = get_recommendation(
                result["classifier_prediction"]
            )

            result["recommendation"] = recommendation

        else:

            result["recommendation"] = {
                "description":
                    "This disease is not present in the training dataset.",
                "advice":
                    "Consult an agricultural expert or upload additional images."
            }

        result["success"] = True

        return result

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)