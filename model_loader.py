from download_models import download_models

download_models()

import torch
import timm
import json
import faiss

import numpy as np

embedding_labels = np.load(
    "model_files/embedding_labels.npy"
)

MODEL_PATH = "model_files/plant_disease_model.pth"

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

class_names = checkpoint["class_names"]

classification_model = timm.create_model(
    "efficientnet_b3",
    pretrained=False,
    num_classes=len(class_names)
)

classification_model.load_state_dict(
    checkpoint["model_state_dict"]
)

classification_model.eval()

index = faiss.read_index(
    "model_files/disease_index.faiss"
)

with open(
    "model_files/class_mapping.json",
    "r"
) as f:
    class_mapping = json.load(f)