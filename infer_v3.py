# infer_v3.py
import os, sys, json, torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2, numpy as np

CKPT="xr_mvp_best.pt"; CALIB="calibration_threshold.json"
DEVICE="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

def load_model():
    ckpt=torch.load(CKPT,map_location=DEVICE)
    name=ckpt.get("model_name","efficientnet_b0")
    if name=="efficientnet_b0":
        m=models.efficientnet_b0(weights=None); m.classifier[1]=nn.Linear(m.classifier[1].in_features,1); target="features.6.0"
    elif name=="efficientnet_b3":
        m=models.efficientnet_b3(weights=None); m.classifier[1]=nn.Linear(m.classifier[1].in_features,1); target="features.8.0"
    elif name=="densenet121":
        m=models.densenet121(weights=None); m.classifier=nn.Linear(m.classifier.in_features,1); target="features.denseblock4"
    else:
        m=models.resnet50(weights=None); m.fc=nn.Linear(m.fc.in_features,1); target="layer4"
    m.load_state_dict(ckpt["model_state"]); m.eval().to(DEVICE)
    mean,std=ckpt["normalize"]; return m,ckpt["img_size"],(mean,std),ckpt["classes"],target

def prep(p,sz,norm):
    mean,std=norm
    tf=transforms.Compose([
        transforms.Grayscale(3), transforms.Resize(int(sz*1.15)),
        transforms.CenterCrop(sz), transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])
    return tf(Image.open(p).convert("L")).unsqueeze(0)

def cam(model,x,layer):
    feats,grads={},{}
    L=dict([*model.named_modules()])[layer]
    h1=L.register_forward_hook(lambda m,i,o: feats.setdefault("y",o))
    h2=L.register_backward_hook(lambda m,gi,go: grads.setdefault("y",go[0]))
    z=model(x).squeeze(1); model.zero_grad(); z.backward(torch.ones_like(z))
    a=feats["y"].detach().cpu().numpy()[0]; g=grads["y"].detach().cpu().numpy()[0]; w=g.mean((1,2))
    m=np.maximum((w[:,None,None]*a).sum(0),0); m=m/(m.max()+1e-6); h1.remove(); h2.remove(); return m

def overlay(img_path, m):
    raw=cv2.imread(img_path); m=cv2.resize(m,(raw.shape[1],raw.shape[0]))
    heat=cv2.applyColorMap((m*255).astype(np.uint8), cv2.COLORMAP_JET)
    return (0.35*heat+0.65*raw).astype(np.uint8)

if __name__=="__main__":
    if len(sys.argv)<2: print("Kullanım: python infer_v3.py <image>"); sys.exit(1)
    img=sys.argv[1]
    model,sz,norm,classes,layer=load_model()
    T,TH=1.0,0.5
    if os.path.exists(CALIB):
        d=json.load(open(CALIB)); T=float(d.get("T",1.0)); TH=float(d.get("threshold",0.5))

    x=prep(img,sz,norm).to(DEVICE)
    with torch.no_grad():
        logit=model(x).squeeze(1).item()
        prob=torch.sigmoid(torch.tensor(logit/T)).item()
    pred=classes[1] if prob>=TH else classes[0]
    print(f"Prediction: {pred} | Prob(PNEUMONIA)={prob:.3f} | Threshold={TH:.2f}")
    vis=overlay(img, cam(model,x,layer)); cv2.imwrite("cam_overlay.png",vis)
    print("Grad-CAM kaydedildi: cam_overlay.png")
