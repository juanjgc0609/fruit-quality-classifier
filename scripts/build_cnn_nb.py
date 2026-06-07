"""Genera y ejecuta notebooks/4_modeling/cnn.ipynb (CNN desde cero + ablation)."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks/4_modeling/cnn.ipynb"

cells = []
M = lambda s: cells.append(new_markdown_cell(s))
C = lambda s: cells.append(new_code_cell(s))

M("""# Fase 4 (parte B) — CNN desde cero
**Proyecto:** FruitVision — Clasificación de Calidad de Frutas

---
CNN entrenada **desde cero**: 3 bloques Conv→BatchNorm→MaxPool + GAP + Dense +
Dropout, con data augmentation, `class_weight` y callbacks.

**Ablation:** entrenamos la CNN con dos variantes de train —**con** y **sin** los
recortes de Mixed (`mixed_seg`)— con idénticos val/test, y guardamos la ganadora.""")

C("""import sys, pathlib, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
ROOT = pathlib.Path.cwd()
while not (ROOT / "src").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, cv2
import matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style="whitegrid")
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from src.config import QUALITY_CLASSES, CNN_IMG_SIZE, MODELS_DIR, FIGURES_DIR, SEED
from src.data.preprocessing import load_manifest
from src.data.paths import load_image_rgb
tf.random.set_seed(SEED); np.random.seed(SEED)
print("TF", tf.__version__, "| img", CNN_IMG_SIZE)""")

M("## 1. Cargar imágenes en memoria (train completo / sin mixed_seg / val / test)")
C("""def load_df(m, size=CNN_IMG_SIZE):
    X=np.zeros((len(m),*size,3),np.float32); keep=np.ones(len(m),bool)
    for i,p in enumerate(m['abs_path']):
        im=load_image_rgb(p)
        if im is None: keep[i]=False; continue
        X[i]=cv2.resize(im,size).astype(np.float32)/255.0
    return X[keep], m['quality_idx'].values[keep]
m_tr=load_manifest('train'); m_trn=m_tr[m_tr['source']!='mixed_seg']
Xtr,ytr   = load_df(m_tr)
Xtrn,ytrn = load_df(m_trn)
Xva,yva   = load_df(load_manifest('val'))
Xte,yte   = load_df(load_manifest('test'))
print(f"train completo={len(Xtr)} | sin mixed_seg={len(Xtrn)} | val={len(Xva)} | test={len(Xte)}")""")

M("## 2. Arquitectura y entrenamiento (función reutilizable)")
C("""def build_cnn():
    aug = models.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.08),
                             layers.RandomZoom(0.10), layers.RandomBrightness(0.10, value_range=(0,1))])
    m = models.Sequential([
        layers.Input((*CNN_IMG_SIZE,3)), aug,
        layers.Conv2D(32,3,padding='same',activation='relu'), layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.Conv2D(64,3,padding='same',activation='relu'), layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.Conv2D(128,3,padding='same',activation='relu'), layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(), layers.Dense(128,activation='relu'), layers.Dropout(0.5),
        layers.Dense(len(QUALITY_CLASSES), activation='softmax')])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

def train_variant(Xt, yt, tag):
    tf.keras.utils.set_random_seed(SEED)
    cw = dict(enumerate(compute_class_weight('balanced', classes=np.unique(yt), y=yt)))
    cbs=[callbacks.EarlyStopping(monitor='val_loss',patience=6,restore_best_weights=True),
         callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=3,min_lr=1e-5)]
    m=build_cnn()
    h=m.fit(Xt,yt,validation_data=(Xva,yva),epochs=40,batch_size=32,
            class_weight=cw,callbacks=cbs,verbose=2)
    p=m.predict(Xte,verbose=0).argmax(1)
    acc=accuracy_score(yte,p); f1=f1_score(yte,p,average='macro')
    print(f"[{tag}] test accuracy={acc:.3f} | f1_macro={f1:.3f}")
    return m,h,acc,f1,p""")

M("## 3. Ablation — CNN con vs sin `mixed_seg`")
C("""cnn_full, h_full, acc_f, f1_f, p_f = train_variant(Xtr,  ytr,  'con mixed_seg')""")
C("""cnn_nomix, h_nomix, acc_n, f1_n, p_n = train_variant(Xtrn, ytrn, 'sin mixed_seg')""")
C("""abl=pd.DataFrame([{'variante':'con mixed_seg','accuracy':acc_f,'f1_macro':f1_f},
                  {'variante':'sin mixed_seg','accuracy':acc_n,'f1_macro':f1_n}]).set_index('variante')
print(abl.round(4).to_string())
best_tag = 'con mixed_seg' if f1_f>=f1_n else 'sin mixed_seg'
cnn, hist, pred = (cnn_full,h_full,p_f) if best_tag=='con mixed_seg' else (cnn_nomix,h_nomix,p_n)
print(f"\\nMejor variante CNN: {best_tag}")
abl.to_csv(MODELS_DIR/"ablation_cnn.csv")""")

M("## 4. Curvas de aprendizaje (variante ganadora)")
C("""h=hist.history
fig,ax=plt.subplots(1,2,figsize=(11,4))
ax[0].plot(h['loss'],label='train'); ax[0].plot(h['val_loss'],label='val'); ax[0].set_title('Pérdida'); ax[0].legend()
ax[1].plot(h['accuracy'],label='train'); ax[1].plot(h['val_accuracy'],label='val'); ax[1].set_title('Accuracy'); ax[1].legend()
plt.tight_layout(); plt.savefig(FIGURES_DIR/"fase4_cnn_curvas.pdf", bbox_inches="tight"); plt.show()""")

M("## 5. Evaluación en test (variante ganadora)")
C("""acc=accuracy_score(yte,pred); f1=f1_score(yte,pred,average='macro')
print(f"CNN ({best_tag}) -> accuracy={acc:.3f} | f1_macro={f1:.3f}\\n")
print(classification_report(yte,pred,target_names=QUALITY_CLASSES,digits=3))
fig,axx=plt.subplots(figsize=(5,4.5))
ConfusionMatrixDisplay(confusion_matrix(yte,pred),display_labels=QUALITY_CLASSES).plot(ax=axx,cmap='Greens',colorbar=False)
axx.set_title(f'CNN ({best_tag}) — Matriz de confusión'); axx.tick_params(axis='x',rotation=30)
plt.tight_layout(); plt.savefig(FIGURES_DIR/"fase4_cnn_confusion.pdf", bbox_inches="tight"); plt.show()""")

M("## 6. Guardado")
C("""cnn.save(MODELS_DIR/"cnn_quality.keras")
pd.DataFrame({'modelo':['CNN'],'variante':[best_tag],'accuracy':[acc],'f1_macro':[f1]}).to_csv(MODELS_DIR/"cnn_metrics.csv", index=False)
print("Modelo guardado en models/saved/cnn_quality.keras")""")

M("➡️ **Siguiente:** Fase 5 — Evaluación comparativa.")

nb = new_notebook(cells=cells)
nb.metadata.kernelspec = {"display_name":"Python (fruit-quality)","language":"python","name":"fruit-quality"}
OUT.parent.mkdir(parents=True, exist_ok=True)
ExecutePreprocessor(timeout=5400, kernel_name="fruit-quality").preprocess(nb, {"metadata": {"path": str(OUT.parent)}})
nbf.write(nb, str(OUT))
print("OK ->", OUT)
