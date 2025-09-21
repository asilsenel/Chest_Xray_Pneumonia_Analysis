# batch_infer_v2.py
# Klasördeki tüm görüntüler için inference (kalibrasyon + threshold)
# CSV çıktı: filepath,prediction,prob_pneumonia

import os, csv, argparse, json
import torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image

CKPT  = "xr_mvp_best.pt"
CALIB = "calibration_threshold.json"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

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
    else:
        m = models.resnet50(weights=None)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, 1)

    m.load_state_dict(ckpt["model_state"])
    m.eval().to(DEVICE)
    mean, std = ckpt["normalize"]
    return m, ckpt["img_size"], (mean, std), ckpt["classes"]

def preprocess(path, img_size, norm):
    mean, std = norm
    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    img = Image.open(path).convert("L")
    return tf(img).unsqueeze(0)

def batch_infer(img_dir, out_csv="batch_results.csv"):
    model, img_size, norm, classes = load_model()

    # Calibration (Temperature + Threshold)
    T, TH = 1.0, 0.5
    if os.path.exists(CALIB):
        d = json.load(open(CALIB))
        T = float(d.get("T", 1.0))
        TH = float(d.get("threshold", 0.5))

    results = []
    total = 0

    for root, _, files in os.walk(img_dir):
        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            fpath = os.path.join(root, fname)
            total += 1
            try:
                x = preprocess(fpath, img_size, norm).to(DEVICE)
                with torch.no_grad():
                    logit = model(x).squeeze(1).item()
                    prob = torch.sigmoid(torch.tensor(logit / T)).item()
                pred = classes[1] if prob >= TH else classes[0]
                results.append([fpath, pred, prob])
            except Exception as e:
                results.append([fpath, "ERROR", str(e)])

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "prediction", "prob_pneumonia"])
        writer.writerows(results)

    ok = sum(1 for r in results if r[1] != "ERROR")
    print(f"✅ İşlenen görüntü: {ok}/{total}. Sonuçlar: {out_csv}")
    if os.path.exists(CALIB):
        print(f"ℹ️ Kalibrasyon kullanıldı → T={T:.3f}, threshold={TH:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="data/test", help="Tahmin yapılacak klasör")
    parser.add_argument("--out", type=str, default="test_results_v2.csv", help="Çıktı CSV yolu")
    args = parser.parse_args()

    batch_infer(args.dir, args.out)
