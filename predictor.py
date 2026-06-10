import torch
import torch.nn as nn
import numpy as np
import math
from PIL import Image
from torchvision import transforms
from model_loader import (
    classification_model,
    checkpoint,
    index,
    class_names,
    embedding_labels,
    class_mapping
)
import timm

feature_model = timm.create_model(
    "efficientnet_b3",
    pretrained=False,
    num_classes=len(class_names)
)

feature_model.load_state_dict(
    checkpoint["model_state_dict"]
)

feature_model.classifier = nn.Identity()

feature_model.eval()

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor()
])

def predict_leaf(image_path):

    image = Image.open(image_path).convert("RGB")

    x = transform(image).unsqueeze(0)

    # Classification
    with torch.no_grad():

        logits = classification_model(x)

        probs = torch.softmax(logits, dim=1)

        confidence = float(torch.max(probs))

        if math.isnan(confidence) or math.isinf(confidence):
            confidence = 0.0

        pred_idx = int(torch.argmax(probs))

        prediction = class_names[pred_idx]

    # Embedding
    with torch.no_grad():

        embedding = feature_model(x)

        embedding = embedding.numpy().astype("float32")

    # Nearest neighbors
    distances, indices = index.search(
        embedding,
        k=5
    )

    nearest_diseases = []

    for idx in indices[0]:

        label_id = int(embedding_labels[idx])

        disease_name = class_mapping[str(label_id)]

        if "___" in disease_name:
            disease_name = disease_name.split("___", 1)[1]

        disease_name = disease_name.replace("_", " ")

        nearest_diseases.append(disease_name)

    # --------------------------------
    # Unknown Disease Detection
    # --------------------------------

    nearest_distance = float(distances[0][0])

    top_disease = nearest_diseases[0]
    agreement = int(nearest_diseases.count(top_disease))

    # Default
    status = "Known Disease"

    if "___" in prediction:
        plant_name, disease_name = prediction.split("___", 1)

        final_prediction = disease_name.replace("_", " ")
    else:
        final_prediction = prediction

    # Random object / non-leaf
    # Suspicious image
    if nearest_distance > 250 and agreement <= 2:
        status = "Unknown Disease"
        final_prediction = "Unknown Disease"

    # Borderline
    elif nearest_distance > 180:
        status = "Possibly Known"

    print(
        "Disease Status:",
        status,
        "| Prediction:",
        final_prediction
    )

    safe_distances = []

    for d in distances[0]:
        value = float(d)

        if math.isnan(value) or math.isinf(value):
            value = 0.0

        safe_distances.append(value)

    if math.isnan(nearest_distance) or math.isinf(nearest_distance):
        nearest_distance = 0.0

    confidence = round(float(confidence), 2)
    nearest_distance = round(float(nearest_distance), 2)
    agreement = int(agreement)    

    return {
        "prediction": final_prediction,
        "classifier_prediction": prediction,
        "confidence": confidence,
        "status": status,
        "nearest_diseases": nearest_diseases,
        "agreement": agreement,
        "all_distances": safe_distances,
        "distance": nearest_distance
    }