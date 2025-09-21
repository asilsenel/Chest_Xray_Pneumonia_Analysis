# Chest_Xray_Pneumonia_Analysis
Interpreting chest x-ray images in case of pneumonia disease


-------
1.1. Ortam / Kurulum

Python: 3.10 (venv)

Paketler: torch, torchvision, numpy, pandas, scikit-learn, opencv-python, pillow, tqdm, matplotlib

Donanım: Apple Silicon (mps) hızlandırma aktif

1.2. Veri Düzeni
data/
 ├─ train/
 │   ├─ NORMAL/
 │   └─ PNEUMONIA/
 ├─ val/
 │   ├─ NORMAL/
 │   └─ PNEUMONIA/
 └─ test/
     ├─ NORMAL/
     └─ PNEUMONIA/


Kaggle “Chest X-Ray Images (Pneumonia)” seti kullanıldı.

Hasta-bazlı sızıntı (leakage) riskini azaltmak için verilen train/val/test bölünmesi korundu.

1.3. Ön işleme (preprocess)

Röntgen gri → 3 kanala kopya (ImageNet ön-eğitimle uyum)

Resize(int(img_size*1.15)) + CenterCrop(img_size) (val/test)

Normalizasyon: ImageNet mean/std
mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]

Data Augment (train)

RandomResizedCrop(scale=(0.9,1.0))

RandomHorizontalFlip(0.5)

RandomRotation(±5°)
(Anatomiyi bozmayacak “hafif” augment’lar.)

1.4. Model

Varsayılan: EfficientNet-B0 (ImageNet ön-eğitimli)

Son katman 1 çıkışlı (logit) → BCEWithLogitsLoss

Class imbalance:

pos_weight = N_neg / N_pos → pozitif sınıf (PNEUMONIA) kaybını daha ağır cezalandırır.

WeightedRandomSampler → minibatch’lerde sınıf temsili dengelendi.

1.5. Eğitim stratejisi

Freeze → Unfreeze: ilk 2 epoch sadece sınıflandırma bloğu eğitildi, sonra tüm katmanlar açıldı.

Optimizasyon: AdamW(lr=3e-4, weight_decay=1e-4)

LR scheduler: ReduceLROnPlateau (val AUROC’a göre)

Early stopping: 5 epoch iyileşme yoksa durdur.

1.6. Kalibrasyon ve Eşik (threshold)

Val set logit’lerinden Temperature Scaling (T) ile olasılıklar kalibre edildi.

Eşik seçimi: grid arama. (v2’de F1 maksimize; v3’te opsiyonel youden/cost var.)

Seçilen T ve threshold calibration_threshold.json içine kaydedildi.

İnfer/batch-infer bu dosyayı otomatik okur.

1.7. Çıktılar / Dosyalar

xr_mvp_best.pt → en iyi val AUROC ağırlıkları (normalize=(mean,std), img_size, sınıf isimleri)

calibration_threshold.json → { "T": ..., "threshold": ... }

batch_infer_v2.py → toplu tahmin CSV’si (ör. test_results_v2.csv)

analyze_results.py → reports_v2/ klasöründe:

confusion_matrix.png

roc_curve.png

pr_curve.png

1.8. Sonuçlar (test)

Son ekran görüntündeki koşuda (kalibrasyon sonrası):

AUROC ≈ 0.97

Accuracy ≈ 0.93

F1 ≈ 0.95

Sınıf bazında:

NORMAL: precision 0.97, recall 0.84
→ Normal vakalarda yanlış alarm (FP) oldukça düşük; bazı normal görüntüler “pnömoni” demiş olabilir ama seviye kabul edilebilir.

PNEUMONIA: precision 0.91, recall 0.98
→ Hasta kaçırma (FN) çok düşük — sizin klinik önceliğinizle uyumlu.

Not: Son koşuda T ≈ 2.08 ve threshold ≈ 0.85 çıktı; bu eşik, FP’yi kontrol altında tutarken PNEUMONIA recall’ını da çok yüksek bırakmış (olasılıklar belirgin ayrılmış).


Kaynak data link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
