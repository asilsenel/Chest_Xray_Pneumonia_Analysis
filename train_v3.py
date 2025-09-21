# train_v3.py
# Geliştirilmiş eğitim:
# - pos_weight + WeightedRandomSampler
# - Freeze→unfreeze
# - ReduceLROnPlateau
# - Temperature Scaling (T)
# - Threshold tuning: f1 / youden / cost (c_fn, c_fp)
# - EfficientNet-B0/B3, DenseNet121, ResNet50 seçenekleri

import os, random, json, argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------- Args ----------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--img_size", type=int, default=384)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--model", type=str, default="efficientnet_b0",
                   choices=["efficientnet_b0", "efficientnet_b3", "densenet121", "resnet50"])
    p.add_argument("--save_path", type=str, default="xr_mvp_best.pt")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    # threshold ve maliyet parametreleri
    p.add_argument("--threshold_objective", type=str, default="cost",
                   choices=["f1", "youden", "cost"],
                   help="FN'i pahalı gördüğünüz için varsayılan 'cost'")
    p.add_argument("--c_fn", type=float, default=5.0, help="FN maliyeti")
    p.add_argument("--c_fp", type=float, default=1.0, help="FP maliyeti")
    return p.parse_args()

# --------------- Utils ----------------
def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def device_of():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def build_model(name: str):
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, 1)
    elif name == "efficientnet_b3":
        m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, 1)
    elif name == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_f = m.classifier.in_features
        m.classifier = nn.Linear(in_f, 1)
    else:
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, 1)
    return m

def make_transforms(img_size):
    train_tf = transforms.Compose([
        transforms.Grayscale(3),
        transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf

def make_loaders(data_dir, img_size, batch_size, num_workers):
    train_tf, eval_tf = make_transforms(img_size)

    def ld(split, tfm, shuffle=False, sampler=None):
        ds = datasets.ImageFolder(os.path.join(data_dir, split), transform=tfm)
        return DataLoader(ds, batch_size=batch_size,
                          shuffle=(shuffle and sampler is None),
                          sampler=sampler, num_workers=num_workers,
                          pin_memory=torch.cuda.is_available()), ds

    base_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    cnt = Counter(base_ds.targets)  # {0: N_normal, 1: N_pneu}
    N_neg, N_pos = cnt[0], cnt[1]
    pos_weight = torch.tensor([N_neg / max(1, N_pos)], dtype=torch.float32)

    weights = [1.0 / N_neg if y == 0 else 1.0 / N_pos for y in base_ds.targets]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader, train_ds = ld("train", train_tf, sampler=sampler)
    val_loader, _ = ld("val", eval_tf)
    test_loader, _ = ld("test", eval_tf)
    return train_loader, val_loader, test_loader, train_ds.classes, pos_weight

@torch.no_grad()
def evaluate(model, loader, device, T=1.0, thr=0.5):
    model.eval()
    ys, probs = [], []
    for x, y in loader:
        x = x.to(device)
        logit = model(x).squeeze(1)
        p = torch.sigmoid(logit / T).cpu().numpy()
        ys.extend(y.numpy()); probs.extend(p)
    ys, probs = np.array(ys), np.array(probs)
    preds = (probs >= thr).astype(int)
    return {
        "acc": accuracy_score(ys, preds),
        "f1": f1_score(ys, preds),
        "auroc": roc_auc_score(ys, probs),
        "ap": average_precision_score(ys, probs),
    }, ys, probs, preds

# ----------- Calibration ------------
class TemperatureScaler(nn.Module):
    def __init__(self): super().__init__(); self.logT = nn.Parameter(torch.zeros(1))
    def forward(self, z): return z / self.logT.exp()

def collect_logits_labels(model, loader, device):
    model.eval()
    L, Y = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            L.append(model(x).squeeze(1).cpu())
            Y.append(y)
    return torch.cat(L), torch.cat(Y)

def fit_temperature(logits, labels):
    ts = TemperatureScaler()
    opt = optim.LBFGS(ts.parameters(), lr=1.0, max_iter=50)
    crit = nn.BCEWithLogitsLoss()
    z = logits.unsqueeze(1); y = labels.float().unsqueeze(1)
    def closure():
        opt.zero_grad(); loss = crit(ts(z), y); loss.backward(); return loss
    opt.step(closure)
    return ts.logT.exp().item()

def tune_threshold(logits, y_true, T=1.0, objective="cost", c_fp=1.0, c_fn=5.0):
    p = torch.sigmoid(logits / T).numpy()
    y = y_true.numpy()
    grid = np.linspace(0.05, 0.95, 19)
    best_t, best_score = 0.5, -1e12
    for t in grid:
        pred = (p >= t).astype(int)
        TP = ((y==1)&(pred==1)).sum()
        FP = ((y==0)&(pred==1)).sum()
        TN = ((y==0)&(pred==0)).sum()
        FN = ((y==1)&(pred==0)).sum()
        if objective == "f1":
            score = f1_score(y, pred)
        elif objective == "youden":
            tpr = TP / max(1, TP+FN)
            fpr = FP / max(1, FP+TN)
            score = tpr - fpr
        else:  # cost
            score = -(c_fn*FN + c_fp*FP)
        if score > best_score:
            best_score, best_t = score, t
    return best_t

# ------------------ Main -------------------
def main():
    args = get_args()
    set_seed(args.seed)
    device = device_of()
    print("Device:", device)

    tr_loader, va_loader, te_loader, classes, pos_w = make_loaders(
        args.data_dir, args.img_size, args.batch_size, args.num_workers
    )
    print("Classes:", classes)

    model = build_model(args.model).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w.to(device))
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

    # Freeze → unfreeze
    for p in model.parameters(): p.requires_grad = False
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters(): p.requires_grad = True
    elif hasattr(model, "fc"):
        for p in model.fc.parameters(): p.requires_grad = True

    best_val, bad, patience, warm = -1, 0, 5, 2
    for epoch in range(1, args.epochs+1):
        model.train()
        if epoch == warm+1:
            for p in model.parameters(): p.requires_grad = True

        pbar = tqdm(tr_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device).float()
            opt.zero_grad()
            loss = crit(model(x).squeeze(1), y)
            loss.backward(); opt.step()
            pbar.set_postfix(loss=float(loss))

        val_m, _, _, _ = evaluate(model, va_loader, device)
        print(f"[VAL] acc={val_m['acc']:.3f} f1={val_m['f1']:.3f} auroc={val_m['auroc']:.3f}")
        sch.step(val_m["auroc"])

        if val_m["auroc"] > best_val:
            best_val = val_m["auroc"]; bad = 0
            torch.save({
                "model_state": model.state_dict(),
                "classes": classes,
                "img_size": args.img_size,
                "normalize": (IMAGENET_MEAN, IMAGENET_STD),
                "model_name": args.model,
            }, args.save_path)
            print("✅ Best model saved")
        else:
            bad += 1
            if bad >= patience:
                print("⏹ Early stopping"); break

    # Best model ile kalibrasyon & threshold
    ckpt = torch.load(args.save_path, map_location=device)
    model = build_model(ckpt["model_name"]).to(device)
    model.load_state_dict(ckpt["model_state"])

    val_logits, val_y = collect_logits_labels(model, va_loader, device)
    T = fit_temperature(val_logits, val_y)
    best_t = tune_threshold(
        val_logits, val_y, T=T,
        objective=args.threshold_objective, c_fp=args.c_fp, c_fn=args.c_fn
    )
    print(f"[Calibration] T={T:.3f}, best threshold={best_t:.2f} (obj={args.threshold_objective})")

    test_m, y_true, y_prob, y_pred = evaluate(model, te_loader, device, T=T, thr=best_t)
    print("[TEST]", test_m)
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=classes))

    with open("calibration_threshold.json", "w") as f:
        json.dump({"T": float(T), "threshold": float(best_t)}, f, indent=2)
    print("✅ Calibration + threshold saved")

if __name__ == "__main__":
    main()
