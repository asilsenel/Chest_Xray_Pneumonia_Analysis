# infer.py
# Tek görüntüden tahmin + Grad-CAM görselleştirme

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import sys

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
        target_layer = "features.6.0"   # EfficientNet-B0 son blok
    elif model_name == "densenet121":
        m = models.densenet121(weights=None)
        in_features = m.classifier.in_features
        m.classifier = nn.Linear(in_features, 1)
        target_layer = "features.denseblock4"
    else:  # resnet50
        m = models.resnet50(weights=None)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, 1)
        target_layer = "layer4"

    m.load_state_dict(ckpt["model_state"])
    m.eval().to(DEVICE)
    return m, ckpt["img_size"], ckpt["normalize"], ckpt["classes"], target_layer

# -----------------------------------------------------
# Görsel ön işleme
# -----------------------------------------------------
def preprocess(img_path, img_size, normalize):
    mean, std = normalize
    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(int(img_size*1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    img = Image.open(img_path).convert("L")
    return tf(img).unsqueeze(0)

# -----------------------------------------------------
# Grad-CAM hesaplama
# -----------------------------------------------------
def grad_cam(model, x, target_layer):
    feats, grads = {}, {}

    def fwd_hook(m, i, o): feats["y"] = o
    def bwd_hook(m, gi, go): grads["y"] = go[0]

    layer = dict([*model.named_modules()])[target_layer]
    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_backward_hook(bwd_hook)

    logits = model(x)
    logit = logits.squeeze(1)
    score = logit
    model.zero_grad()
    score.backward(torch.ones_like(score))

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

# -----------------------------------------------------
# Main
# -----------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python infer.py <görüntü_dosyası>")
        sys.exit(1)

    img_path = sys.argv[1]
    model, img_size, norm, classes, target_layer = load_model()
    x = preprocess(img_path, img_size, norm).to(DEVICE)

    with torch.no_grad():
        logit = model(x).squeeze(1)
        prob = torch.sigmoid(logit).item()

    pred = classes[1] if prob >= 0.5 else classes[0]
    print(f"Prediction: {pred}  |  Prob(PNEUMONIA)={prob:.3f}")

    cam = grad_cam(model, x, target_layer)
    vis = overlay_cam(img_path, cam)
    out_path = "cam_overlay.png"
    cv2.imwrite(out_path, vis)
    print(f"Grad-CAM kaydedildi: {out_path}")
