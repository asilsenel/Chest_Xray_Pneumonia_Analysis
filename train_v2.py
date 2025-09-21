# train_v2.py
# Pneumonia vs Normal X-ray sınıflandırıcı
# - Class imbalance: pos_weight + (opsiyonel) weighted sampler
# - Freeze→unfreeze
# - LR scheduler (ReduceLROnPlateau)
# - Early stopping
# - Validation üstünde Temperature Scaling + Threshold tuning
# Çıktılar:
#   - xr_mvp_best.pt
#   - calibration_threshold.json

import os, random, json
from pathlib import Path
import argparse
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, classification_report, average_precision_score
)
from tqdm import tqdm
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# -------------------
# Argümanlar
# -------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--img_size", type=int, default=384)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--model", type=str, default="efficientnet_b0",
                   choices=["efficientnet_b0", "densenet121", "resnet50"])
    p.add_argument("--save_path", type=str, default="xr_mvp_best.pt")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

# -------------------
# Utils
# -------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def build_model(name: str):
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, 1)
    elif name == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_features = m.classifier.in_features
        m.classifier = nn.Linear(in_features, 1)
    else:  # resnet50
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, 1)
    return m

def make_dataloaders(data_dir, img_size, batch_size, num_workers):
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    def ld(split, tfm, shuffle=False, sampler=None):
        ds = datasets.ImageFolder(os.path.join(data_dir, split), transform=tfm)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ), ds

    # training set stats for imbalance
    base_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    cnt = Counter(base_ds.targets)  # {0: N_normal, 1: N_pneumonia}
    N_neg, N_pos = cnt[0], cnt[1]
    pos_weight = torch.tensor([N_neg / max(1, N_pos)], dtype=torch.float32)

    # Weighted sampler (isteğe bağlı; küçük veri setlerinde faydalı)
    weights = [1.0 / N_neg if y == 0 else 1.0 / N_pos for y in base_ds.targets]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader, train_ds = ld("train", train_tf, sampler=sampler)
    val_loader,   _        = ld("val",   eval_tf)
    test_loader,  _        = ld("test",  eval_tf)

    classes = train_ds.classes  # ['NORMAL', 'PNEUMONIA']
    return train_loader, val_loader, test_loader, classes, pos_weight

@torch.no_grad()
def evaluate(model, loader, device, T=1.0, threshold=0.5):
    model.eval()
    y_true, y_prob = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x).squeeze(1)
        prob = torch.sigmoid(logits / T)
        y_true.extend(y.tolist())
        y_prob.extend(prob.cpu().numpy().tolist())
    y_true = np.array(y_true); y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auroc": roc_auc_score(y_true, y_prob),
        "ap": average_precision_score(y_true, y_prob),
    }, y_true, y_prob, y_pred

# -------------------
# Calibration utils
# -------------------
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros(1))  # T=1

    def forward(self, logits):
        return logits / self.log_T.exp()

def fit_temperature(logits, labels):
    ts = TemperatureScaler()
    optimizer = optim.LBFGS(ts.parameters(), lr=1.0, max_iter=50)
    bce = nn.BCEWithLogitsLoss()

    logits = logits.unsqueeze(1)
    labels = labels.float().unsqueeze(1)

    def closure():
        optimizer.zero_grad()
        loss = bce(ts(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return ts.log_T.exp().item()

def collect_logits_labels(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x).squeeze(1).detach().cpu()
            all_logits.append(logits)
            all_y.append(y)
    return torch.cat(all_logits), torch.cat(all_y)

def tune_threshold(logits, y_true, T=1.0):
    from sklearn.metrics import f1_score
    prob = torch.sigmoid(logits / T).numpy()
    y_true = y_true.numpy()
    ts = np.linspace(0.1, 0.9, 17)
    best_t, best_f1 = 0.5, -1
    for t in ts:
        pred = (prob >= t).astype(int)
        f1 = f1_score(y_true, pred)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

# -------------------
# Main
# -------------------
def main():
    args = get_args()
    set_seed(args.seed)
    device = get_device()
    print("Device:", device)

    train_loader, val_loader, test_loader, classes, pos_weight = make_dataloaders(
        args.data_dir, args.img_size, args.batch_size, args.num_workers
    )
    print("Classes:", classes)

    model = build_model(args.model).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    # Freeze backbone first 2 epochs
    for p in model.parameters(): p.requires_grad = False
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters(): p.requires_grad = True
    elif hasattr(model, "fc"):
        for p in model.fc.parameters(): p.requires_grad = True

    best_val = -1
    patience, bad_epochs = 5, 0
    head_warmup_epochs = 2

    for epoch in range(1, args.epochs + 1):
        model.train()
        if epoch == head_warmup_epochs + 1:
            for p in model.parameters(): p.requires_grad = True

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device).float()
            optimizer.zero_grad()
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(loss))

        val_metrics, _, _, _ = evaluate(model, val_loader, device)
        print(f"[VAL] acc={val_metrics['acc']:.3f} f1={val_metrics['f1']:.3f} auroc={val_metrics['auroc']:.3f}")
        scheduler.step(val_metrics["auroc"])

        if val_metrics["auroc"] > best_val:
            best_val = val_metrics["auroc"]
            torch.save({
                "model_state": model.state_dict(),
                "classes": classes,
                "img_size": args.img_size,
                "normalize": (IMAGENET_MEAN, IMAGENET_STD),   # ✔ iki elemanlı (mean, std)
                "model_name": args.model
            }, args.save_path)
            print("✅ Best model saved")
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("⏹ Early stopping")
                break

    # Best modeli yükle
    ckpt = torch.load(args.save_path, map_location=device)
    model = build_model(ckpt["model_name"]).to(device)
    model.load_state_dict(ckpt["model_state"])

    # Kalibrasyon + threshold tuning (validation set)
    val_logits, val_y = collect_logits_labels(model, val_loader, device)
    T = fit_temperature(val_logits, val_y)
    best_t = tune_threshold(val_logits, val_y, T=T)
    print(f"[Calibration] T={T:.3f}, best threshold={best_t:.2f}")

    # Test
    test_metrics, y_true, y_prob, y_pred = evaluate(model, test_loader, device, T=T, threshold=best_t)
    print("[TEST]", test_metrics)
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=classes))

    # Kalibrasyon dosyası
    with open("calibration_threshold.json", "w") as f:
        json.dump({"T": float(T), "threshold": float(best_t)}, f, indent=2)
    print("✅ Calibration + threshold kaydedildi")

if __name__ == "__main__":
    main()
