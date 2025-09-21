# batch_infer_v3.py
import os, csv, json, torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image

CKPT="xr_mvp_best.pt"; CAL="calibration_threshold.json"
DEVICE="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

def load_model():
    ckpt=torch.load(CKPT,map_location=DEVICE); name=ckpt.get("model_name","efficientnet_b0")
    if name=="efficientnet_b0":
        m=models.efficientnet_b0(weights=None); m.classifier[1]=nn.Linear(m.classifier[1].in_features,1)
    elif name=="efficientnet_b3":
        m=models.efficientnet_b3(weights=None); m.classifier[1]=nn.Linear(m.classifier[1].in_features,1)
    elif name=="densenet121":
        m=models.densenet121(weights=None); m.classifier=nn.Linear(m.classifier.in_features,1)
    else:
        m=models.resnet50(weights=None); m.fc=nn.Linear(m.fc.in_features,1)
    m.load_state_dict(ckpt["model_state"]); m.eval().to(DEVICE)
    mean,std=ckpt["normalize"]; return m,ckpt["img_size"],(mean,std),ckpt["classes"]

def prep(path,sz,norm):
    mean,std=norm
    tf=transforms.Compose([
        transforms.Grayscale(3), transforms.Resize(int(sz*1.15)),
        transforms.CenterCrop(sz), transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])
    return tf(Image.open(path).convert("L")).unsqueeze(0)

def run(dir_path, out_csv="results.csv"):
    model,sz,norm,classes=load_model()
    T,TH=1.0,0.5
    if os.path.exists(CAL):
        d=json.load(open(CAL)); T=float(d.get("T",1.0)); TH=float(d.get("threshold",0.5))

    rows=[]; total=0
    for root,_,files in os.walk(dir_path):
        for f in files:
            if not f.lower().endswith((".jpg",".jpeg",".png")): continue
            p=os.path.join(root,f); total+=1
            try:
                x=prep(p,sz,norm).to(DEVICE)
                with torch.no_grad():
                    logit=model(x).squeeze(1).item()
                    prob=float(torch.sigmoid(torch.tensor(logit/T)).item())
                pred=classes[1] if prob>=TH else classes[0]
                rows.append([p,pred,prob])
            except Exception as e:
                rows.append([p,"ERROR",str(e)])
    with open(out_csv,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["filepath","prediction","prob_pneumonia"]); w.writerows(rows)
    ok=sum(1 for r in rows if r[1]!="ERROR")
    print(f"✅ İşlenen: {ok}/{total}. Çıktı: {out_csv} | T={T:.3f}, thr={TH:.2f}")

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--dir",default="data/test"); ap.add_argument("--out",default="test_results_v3.csv")
    a=ap.parse_args(); run(a.dir, a.out)
