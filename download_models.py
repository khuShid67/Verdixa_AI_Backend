import os
import gdown

MODEL_DIR = "model_files"
os.makedirs(MODEL_DIR, exist_ok=True)

FILES = {
    "class_mapping.json": "1uPJhBGapLtyQ2Lpcmrm21WGR5Qo56Fp9",
    "disease_embeddings.npy": "1WdAd21qZ3CiM1jdVJnN1CQwBvq0lfuLw",
    "disease_index.faiss": "1UsQUtNHZ2DbToa7BoMDDqVPp5SHB-QJh",
    "embedding_labels.npy": "1M_FaqzaMWPmjJr7OYLLNZDDeYcTyvAwn",
    "leaf_detector_v2.pth": "1WWj6ZzWo6VXfEyfa5I09qohRTougNT_0",
    "plant_disease_model.pth": "1KSW-8QGdSNlnt0mrssqzpKE74RkjBI0e",
}


def download_file(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"

    if not os.path.exists(output_path):
        print(f"Downloading {os.path.basename(output_path)}...")
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"{os.path.basename(output_path)} already exists")


def download_models():
    for filename, file_id in FILES.items():
        output_path = os.path.join(MODEL_DIR, filename)
        download_file(file_id, output_path)

    print("All model files are ready.")


if __name__ == "__main__":
    download_models()