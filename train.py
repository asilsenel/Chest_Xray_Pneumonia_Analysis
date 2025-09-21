# train.py
# Binary sınıflandırma: NORMAL (0) vs PNEUMONIA (1)
# PyTorch EfficientNet-B0 tabanlı; en iyi model validation AUROC'a göre kaydedilir.

import os
import random
from pathlib import Path
import argparse
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm

# ---------------------------------------------------------
# Argümanlar
# ---------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data", help="train/ val/ test/ kökü")
    p.add_argument("--img_size", type=int, default=384)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--model", type=str, default="efficientnet_b0",
                   choices=["efficientnet_b0", "densenet121", "resnet50"])
    p.add_argument("--save_path", type=str, default="xr_mvp_best.pt")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

# ---------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon (Metal)
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
    # Röntgen tek kanallı olduğundan 3 kanala kopyalayıp ImageNet normalize ediyoruz
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485]*3, [0.456]*3, [0.229]*3)  # (mean, std) per channel
    ])
    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485]*3, [0.456]*3, [0.229]*3)
    ])

    def ld(split, tfm, shuffle=False):
        ds = datasets.ImageFolder(os.path.join(data_dir, split), transform=tfm)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=torch.cuda.is_available()), ds

    train_loader, train_ds = ld("train", train_tf, shuffle=True)
    val_loader, val_ds = ld("val", eval_tf, shuffle=False)
    test_loader, test_ds = ld("test", eval_tf, shuffle=False)

    # Sınıf isimleri ve dağılım bilgisi
    classes = train_ds.classes  # ['NORMAL', 'PNEUMONIA'] bekleniyor
    return train_loader, val_loader, test_loader, classes

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).float()
        logits = model(x).squeeze(1)
        prob = torch.sigmoid(logits)
        y_true.extend(y.cpu().tolist())
        y_prob.extend(prob.cpu().tolist())

    # Threshold = 0.5 ile özet metrikler
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
    metrics = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, y_prob))
    }
    return metrics

# ---------------------------------------------------------
# Eğitim
# ---------------------------------------------------------
def main():
    args = get_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, classes = make_dataloaders(
        args.data_dir, args.img_size, args.batch_size, args.num_workers
    )
    print("Classes:", classes)

    model = build_model(args.model).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # MPS'te autocast desteklenmiyor; sadece CUDA'da açıyoruz.
    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val = -1.0
    patience, bad_epochs = 5, 0  # erken durdurma için basit sabır

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        epoch_loss = 0.0

        for x, y in pbar:
            x = x.to(device)
            y = y.to(device).float()

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(x).squeeze(1)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x).squeeze(1)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

            epoch_loss += float(loss)
            pbar.set_postfix(loss=float(loss))

        # Validation
        val_metrics = evaluate(model, val_loader, device)
        print(f"[VAL] loss={epoch_loss/len(train_loader):.4f} "
              f"acc={val_metrics['acc']:.3f} f1={val_metrics['f1']:.3f} auroc={val_metrics['auroc']:.3f}")

        # Checkpoint
        score = val_metrics["auroc"]
        if score > best_val:
            best_val = score
            torch.save({
                "model_state": model.state_dict(),
                "classes": classes,
                "img_size": args.img_size,
                "normalize": ([0.485, 0.456, 0.229], [0.229, 0.224, 0.225]),
                "model_name": args.model
            }, args.save_path)
            print(f"✅ Best model saved to {args.save_path} (val AUROC={best_val:.3f})")
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("⏹ Early stopping (no val AUROC improvement).")
                break

    # Test
    # En iyi ağırlıklarla yeniden yükleyelim
    if Path(args.save_path).exists():
        ckpt = torch.load(args.save_path, map_location=device)
        model = build_model(ckpt.get("model_name", args.model)).to(device)
        model.load_state_dict(ckpt["model_state"])

    test_metrics = evaluate(model, test_loader, device)
    print(f"[TEST] acc={test_metrics['acc']:.3f} f1={test_metrics['f1']:.3f} auroc={test_metrics['auroc']:.3f}")

    # Sonuçları json olarak da bırak
    with open("last_metrics.json", "w") as f:
        json.dump({
            "val_best_auroc": best_val,
            "test": test_metrics
        }, f, indent=2)
    print("🔎 Metrikler kaydedildi: last_metrics.json")

if __name__ == "__main__":
    main()
