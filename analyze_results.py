# analyze_results.py
import argparse, csv, os, re
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

def infer_label_from_path(path):
    # data/test/NORMAL/... veya data/test/PNEUMONIA/...
    if re.search(r"/PNEUMONIA/", path, re.IGNORECASE):
        return 1
    return 0

def main(csv_path, out_dir="reports"):
    os.makedirs(out_dir, exist_ok=True)
    y_true, y_prob, y_pred = [], [], []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row["filepath"]
            prob = float(row["prob_pneumonia"])
            pred = 1 if row["prediction"].strip().upper() == "PNEUMONIA" else 0
            y_true.append(infer_label_from_path(path))
            y_prob.append(prob)
            y_pred.append(pred)

    y_true = np.array(y_true); y_prob = np.array(y_prob); y_pred = np.array(y_pred)

    # Metrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=["NORMAL","PNEUMONIA"]))
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    print(f"ROC-AUC: {auc:.3f}  |  Average Precision (PR-AUC): {ap:.3f}")

    # Confusion matrix görseli
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xticks([0,1], ["NORMAL","PNEUMONIA"])
    plt.yticks([0,1], ["NORMAL","PNEUMONIA"])
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=160)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=160)
    plt.close()

    # Precision-Recall
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_curve.png"), dpi=160)
    plt.close()

    print(f"✅ Raporlar kaydedildi: {out_dir}/ (confusion_matrix.png, roc_curve.png, pr_curve.png)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="test_results.csv")
    p.add_argument("--out", type=str, default="reports")
    args = p.parse_args()
    main(args.csv, args.out)
