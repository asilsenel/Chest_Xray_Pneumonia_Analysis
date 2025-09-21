# app.py
import io, os, base64, json
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ---- Konfig ----
CKPT_PATH = os.getenv("XR_CKPT", "xr_mvp_best.pt")
CALIB_PATH = os.getenv("XR_CALIB", "calibration_threshold.json")
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

app = FastAPI(title="Chest X-Ray Pneumonia API", version="v1")
# Kökte index.html
@app.get("/", response_class=FileResponse)
def root():
    return FileResponse("index.html")   # index.html proje kökünde

# ---- Model yükleme ----
def build_model(name: str):
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, 1)
        target_layer = "features.6.0"
    elif name == "efficientnet_b3":
        m = models.efficientnet_b3(weights=None)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, 1)
        target_layer = "features.8.0"
    elif name == "densenet121":
        m = models.densenet121(weights=None)
        in_f = m.classifier.in_features
        m.classifier = nn.Linear(in_f, 1)
        target_layer = "features.denseblock4"
    else:  # resnet50
        m = models.resnet50(weights=None)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, 1)
        target_layer = "layer4"
    return m, target_layer

def load_assets():
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model, target_layer = build_model(ckpt.get("model_name", "efficientnet_b0"))
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(DEVICE)

    img_size = ckpt["img_size"]
    mean, std = ckpt["normalize"]
    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    classes = ckpt["classes"]

    T, TH = 1.0, 0.5
    if os.path.exists(CALIB_PATH):
        d = json.load(open(CALIB_PATH))
        T = float(d.get("T", 1.0))
        TH = float(d.get("threshold", 0.5))

    return model, tf, classes, T, TH, target_layer

MODEL, TF, CLASSES, TEMP_T, THRESH, TARGET_LAYER = load_assets()

# ---- Yardımcılar ----
def to_tensor(file_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("L")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Geçersiz görüntü: {e}")
    x = TF(img).unsqueeze(0).to(DEVICE)
    return x

def predict_proba(x: torch.Tensor):
    with torch.no_grad():
        logit = MODEL(x).squeeze(1).item()
        prob = torch.sigmoid(torch.tensor(logit / TEMP_T)).item()
    return float(prob)

def make_cam(x: torch.Tensor):
    # Grad-CAM (basit, son blok)
    feats, grads = {}, {}
    layer = dict([*MODEL.named_modules()])[TARGET_LAYER]

    def fwd_hook(_m, _i, o): feats["y"] = o
    def bwd_hook(_m, _gi, go): grads["y"] = go[0]; return None
    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook) if hasattr(layer, "register_full_backward_hook") else layer.register_backward_hook(bwd_hook)

    logits = MODEL(x).squeeze(1)
    MODEL.zero_grad(set_to_none=True)
    logits.backward(torch.ones_like(logits))

    act = feats["y"].detach().cpu().numpy()[0]
    grad = grads["y"].detach().cpu().numpy()[0]
    w = grad.mean(axis=(1, 2))
    cam = (w[:, None, None] * act).sum(axis=0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)

    h1.remove(); h2.remove()
    # CAM'i base64 PNG olarak döndür
    import cv2, numpy as np
    cam_img = (cam * 255).astype("uint8")
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    _, buf = cv2.imencode(".png", cam_img)
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return b64

# ---- Endpoints ----
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "classes": CLASSES,
        "temperature": TEMP_T,
        "threshold": THRESH
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    x = to_tensor(data)
    prob = predict_proba(x)
    label = CLASSES[1] if prob >= THRESH else CLASSES[0]
    return JSONResponse({
        "label": label,
        "prob_pneumonia": round(prob, 4),
        "threshold": THRESH,
        "temperature": TEMP_T
    })

@app.post("/predict-cam")
async def predict_cam(file: UploadFile = File(...), overlay: Optional[bool] = False):
    # CAM'i ayrı döndürüyoruz; istersen client tarafı overlay yapabilir.
    data = await file.read()
    x = to_tensor(data)
    prob = predict_proba(x)
    label = CLASSES[1] if prob >= THRESH else CLASSES[0]
    cam_b64 = make_cam(x)
    return JSONResponse({
        "label": label,
        "prob_pneumonia": round(prob, 4),
        "threshold": THRESH,
        "temperature": TEMP_T,
        "grad_cam_png_b64": cam_b64
    })
