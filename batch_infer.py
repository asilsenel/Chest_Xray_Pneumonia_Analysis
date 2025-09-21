# batch_infer.py
# Tüm klasörü tara, tahminleri CSV olarak kaydet

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import csv
import argparse

CKPT = "xr_mvp_best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# -----------------------------------------------------
# Model yükleme
# -----------------------------------------------------
def load_model():
    ckpt = torch.load(CKPT, map_location=DEVICE)
    model_name = ckpt.get("model_name", "efficientnet_b0")

    if model_name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, 1)
    elif model_name == "densenet121":
        m = models.densenet121(weights=None)
        in_features = m.classifier.in_features
        m.classifier = nn.Linear(in_features, 1)
    else:  # resnet50
        m = models.resnet50(weights=None)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, 1)

    m.load_state_dict(ckpt["model_state"])
    m.eval().to(DEVICE)
    return m, ckpt["img_size"], ckpt["normalize"], ckpt["classes"]

# -----------------------------------------------------
# Görsel ön işleme
# -----------------------------------------------------
def preprocess(img_path, img_size, normalize):
    mean, std = normalize
    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    img = Image.open(img_path).convert("L")
    return tf(img).unsqueeze(0)

# -----------------------------------------------------
# Batch inference
# -----------------------------------------------------
def batch_infer(img_dir, out_csv="batch_results.csv"):
    model, img_size, norm, classes = load_model()

    results = []
    for root, _, files in os.walk(img_dir):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                fpath = os.path.join(root, fname)
                x = preprocess(fpath, img_size, norm).to(DEVICE)
                with torch.no_grad():
                    logit = model(x).squeeze(1)
                    prob = torch.sigmoid(logit).item()
                pred = classes[1] if prob >= 0.5 else classes[0]
                results.append([fpath, pred, prob])

    # CSV'ye yaz
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "prediction", "prob_pneumonia"])
        writer.writerows(results)

    print(f"✅ {len(results)} görüntü işlendi. Sonuçlar kaydedildi: {out_csv}")

# -----------------------------------------------------
# Main
# -----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="data/test", help="Tahmin yapılacak klasör")
    parser.add_argument("--out", type=str, default="batch_results.csv", help="Çıktı CSV dosyası")
    args = parser.parse_args()

    batch_infer(args.dir, args.out)
