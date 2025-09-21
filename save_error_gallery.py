# save_error_gallery.py  (FIXED)
import os, csv, json, torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np, cv2, pathlib

CKPT = "xr_mvp_best.pt"
CAL  = "calibration_threshold.json"
OUT  = "error_gallery"
os.makedirs(OUT, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

def load():
    ckpt = torch.load(CKPT, map_input=DEVICE) if hasattr(torch.load, "map_input") else torch.load(CKPT, map_location=DEVICE)
    # Model mimarisi: EfficientNet-B0 varsayımı; başka bir mimari kullandıysan target katmanı aşağıda değiştir.
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
    m.load_state_dict(ckpt["model_state"])
    m.eval().to(DEVICE)
    mean, std = ckpt["normalize"]; sz = ckpt["img_size"]; classes = ckpt["classes"]
    return m, sz, (mean, std), classes

def prep(p, sz, norm):
    mean, std = norm
    tf = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(int(sz * 1.15)),
        transforms.CenterCrop(sz),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return tf(Image.open(p).convert("L")).unsqueeze(0)

def grad_cam(model, x, target_layer="features.6.0"):
    feats, grads = {}, {}

    layer = dict([*model.named_modules()])[target_layer]

    def fwd_hook(_m, _i, o):
        feats["y"] = o

    def bwd_hook(_m, _gi, go):
        # go tuple; ilk eleman target gradient
        grads["y"] = go[0]
        # ÖNEMLİ: hiçbir şey döndürme (None)
        return None

    h1 = layer.register_forward_hook(fwd_hook)
    # yeni API daha güvenli:
    if hasattr(layer, "register_full_backward_hook"):
        h2 = layer.register_full_backward_hook(bwd_hook)
    else:
        h2 = layer.register_backward_hook(bwd_hook)

    # forward + backward
    logits = model(x).squeeze(1)
    model.zero_grad(set_to_none=True)
    logits.backward(torch.ones_like(logits))

    act  = feats["y"].detach().cpu().numpy()[0]
    grad = grads["y"].detach().cpu().numpy()[0]
    w = grad.mean(axis=(1, 2))
    cam = np.maximum((w[:, None, None] * act).sum(axis=0), 0)
    cam = cam / (cam.max() + 1e-6)

    h1.remove(); h2.remove()
    return cam

def overlay(p, cam):
    raw = cv2.imread(p)
    cam = cv2.resize(cam, (raw.shape[1], raw.shape[0]))
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return (0.35 * heat + 0.65 * raw).astype(np.uint8)

def true_label_from_path(p):
    return "PNEUMONIA" if "/PNEUMONIA/" in p.upper() else "NORMAL"

def main(csv_path="test_results_v3.csv"):
    model, sz, norm, _ = load()
    T, TH = 1.0, 0.5
    if os.path.exists(CAL):
        d = json.load(open(CAL))
        T = float(d.get("T", 1.0))
        TH = float(d.get("threshold", 0.5))

    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            p = row["filepath"]; pred = row["prediction"].upper()
            if pred == "ERROR":  # hatalı dosya
                continue
            true = true_label_from_path(p)
            if pred != true:     # sadece yanlışları kaydet
                x = prep(p, sz, norm).to(DEVICE)
                with torch.no_grad():
                    logit = model(x).squeeze(1).item()
                    prob = float(torch.sigmoid(torch.tensor(logit / T)).item())
                cam = grad_cam(model, x)
                vis = overlay(p, cam)
                tag = "FP" if (true == "NORMAL" and pred == "PNEUMONIA") else "FN"
                out = os.path.join(OUT, f"{tag}__{pathlib.Path(p).stem}__prob{prob:.2f}.png")
                cv2.imwrite(out, vis)

    print(f"✅ Hata görselleri {OUT}/ içine kaydedildi.")

if __name__ == "__main__":
    main()
