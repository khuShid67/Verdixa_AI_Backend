import torch
import timm
from PIL import Image
from torchvision import transforms
from download_models import download_models

download_models()
DEVICE = "cpu"

checkpoint = torch.load(
    "model_files/leaf_detector_v2.pth",
    map_location=DEVICE
)

leaf_model = timm.create_model(
    "efficientnet_b2",
    pretrained=False,
    num_classes=2
)

leaf_model.load_state_dict(
    checkpoint["model_state_dict"]
)

leaf_model.eval()

classes = checkpoint["class_names"]

transform = transforms.Compose([
    transforms.Resize((260,260)),
    transforms.ToTensor()
])

def is_leaf(image_path):

    image = Image.open(image_path).convert("RGB")

    x = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = leaf_model(x)

        probs = torch.softmax(outputs, dim=1)

        confidence = float(torch.max(probs)) * 100

        pred_idx = int(torch.argmax(probs))

    prediction = classes[pred_idx]

    return prediction, confidence