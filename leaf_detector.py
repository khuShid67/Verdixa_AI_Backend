import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

DEVICE = torch.device("cpu")


# -------------------------
# Model
# -------------------------
class LeafCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -------------------------
# Load model
# -------------------------
checkpoint = torch.load(
    "model_files/leaf_detector_v2.pth",
    map_location=DEVICE
)

leaf_model = LeafCNN().to(DEVICE)
leaf_model.load_state_dict(checkpoint)
leaf_model.eval()


# -------------------------
# FIX: define classes manually
# -------------------------
classes = ["Leaf", "Non_Leaf"]   # ⚠️ adjust based on your training order


# -------------------------
# Transform
# -------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


# -------------------------
# Prediction
# -------------------------
def is_leaf(image_path):

    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = leaf_model(x)
        probs = torch.softmax(outputs, dim=1)

        confidence = float(torch.max(probs)) * 100
        pred_idx = int(torch.argmax(probs))

    prediction = classes[pred_idx]

    return prediction, confidence