# infer_v2.py
# Tek görüntü tahmini (kalibrasyon + threshold) + Grad-CAM

import os, sys, json
import torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

CKPT = "xr_mvp_best.pt"
CALIB = "calibration_threshold.json"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

def load_model():
    ckpt = torch.load(CKPT, map_location=DEVICE)
    model_name = ckpt.get("model_name", "efficientnet_b0")

    if model_name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, 1)
        target_layer = "features.6.0"
    elif model_name == "densenet121":
        m = models.densenet121(weights=None)
        in_features = m.classifier.in_features
        m.classifier = nn.Linear(in_features, 1)
        target_layer = "features.denseblock4"
    else:
        m = models.resnet50(weights=None)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, 1)
        target_layer = "layer4"

    m.load_state_dict(ckpt["model_state"])
    m.eval().to(DEVICE)
    mean, std = ckpt["normalize"]
    return m, ckpt["img_size"], (mean, std), ckpt["classes"], target_layer

def preprocess(img_path, img_size, norm):
    mean, std = norm
    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    img = Image.open(img_path).convert("L")
    return tf(img).unsqueeze(0)

def grad_cam(model, x, target_layer):
    feats, grads = {}, {}
    def fwd_hook(m, i, o): feats["y"] = o
    def bwd_hook(m, gi, go): grads["y"] = go[0]
    layer = dict([*model.named_modules()])[target_layer]
    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_backward_hook(bwd_hook)

    logits = model(x).squeeze(1)
    model.zero_grad()
    logits.backward(torch.ones_like(logits))

    act = feats["y"].detach().cpu().numpy()[0]
    grad = grads["y"].detach().cpu().numpy()[0]
    weights = grad.mean(axis=(1, 2))
    cam = np.maximum((weights[:, None, None] * act).sum(axis=0), 0)
    cam = cam / (cam.max() + 1e-6)
    h1.remove(); h2.remove()
    return cam

def overlay_cam(img_path, cam):
    raw = cv2.imread(img_path)
    cam = cv2.resize(cam, (raw.shape[1], raw.shape[0]))
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    out = (0.35 * heat + 0.65 * raw).astype(np.uint8)
    return out

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python infer_v2.py <image_path>")
        sys.exit(1)
    img_path = sys.argv[1]

    model, img_size, norm, classes, target_layer = load_model()

    T, TH = 1.0, 0.5
    if os.path.exists(CALIB):
        d = json.load(open(CALIB))
        T = float(d.get("T", 1.0))
        TH = float(d.get("threshold", 0.5))

    x = preprocess(img_path, img_size, norm).to(DEVICE)
    with torch.no_grad():
        logit = model(x).squeeze(1).item()
        prob = torch.sigmoid(torch.tensor(logit / T)).item()

    pred = classes[1] if prob >= TH else classes[0]
    print(f"Prediction: {pred} | Prob(PNEUMONIA)={prob:.3f} | Threshold={TH:.2f}")

    cam = grad_cam(model, x, target_layer)
    vis = overlay_cam(img_path, cam)
    cv2.imwrite("cam_overlay.png", vis)
    print("Grad-CAM kaydedildi: cam_overlay.png")
