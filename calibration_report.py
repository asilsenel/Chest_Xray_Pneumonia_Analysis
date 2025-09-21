# calibration_report.py
import csv, numpy as np, matplotlib.pyplot as plt, argparse, os

def reliability_curve(y_true, y_prob, bins=10):
    y_true=np.asarray(y_true); y_prob=np.asarray(y_prob)
    edges=np.linspace(0.0,1.0,bins+1); ece=0.0
    confs=[]; accs=[]; cnts=[]
    for i in range(bins):
        lo,hi=edges[i],edges[i+1]
        idx=(y_prob>=lo)&(y_prob<hi)
        if idx.sum()==0: confs.append((lo+hi)/2); accs.append(np.nan); cnts.append(0); continue
        conf=y_prob[idx].mean(); acc=(y_true[idx].mean())  # positive-class accuracy
        confs.append(conf); accs.append(acc); cnts.append(idx.sum())
        ece += np.abs(acc-conf)*idx.mean()
    return np.array(confs), np.array(accs), ece

def main(csv_path, out_dir="reports_calib"):
    os.makedirs(out_dir,exist_ok=True)
    y_true=[]; y_prob=[]
    with open(csv_path) as f:
        r=csv.DictReader(f)
        for row in r:
            if row["prediction"]=="ERROR": continue
            y_true.append(1 if "/PNEUMONIA/" in row["filepath"].upper() else 0)
            y_prob.append(float(row["prob_pneumonia"]))
    conf, acc, ece = reliability_curve(y_true, y_prob, bins=12)
    plt.figure();
    plt.plot([0,1],[0,1],"--",label="perfect")
    plt.plot(conf, acc, marker="o", label=f"ECE={ece:.03f}")
    plt.xlabel("Confidence"); plt.ylabel("Empirical accuracy (positive)"); plt.title("Reliability diagram")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out_dir,"reliability.png"),dpi=160); plt.close()
    print(f"✅ reliability.png kaydedildi. ECE={ece:.3f}")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--csv",default="test_results_v3.csv"); ap.add_argument("--out",default="reports_calib")
    a=ap.parse_args(); main(a.csv, a.out)
