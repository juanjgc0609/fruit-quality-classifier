"""Genera y ejecuta notebooks/4_modeling/cnn.ipynb (CNN desde cero)."""
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
CNN entrenada **desde cero** (sin transfer learning): 3 bloques
Conv→BatchNorm→MaxPool + GlobalAveragePooling + Dense + Dropout. Con data
augmentation, `class_weight` y callbacks (`EarlyStopping`, `ReduceLROnPlateau`).""")

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

M("## 1. Cargar imágenes en memoria")
C("""def load_split(split, size=CNN_IMG_SIZE):
    m = load_manifest(split); X = np.zeros((len(m), *size, 3), np.float32); keep = np.ones(len(m), bool)
    for i, p in enumerate(m['abs_path']):
        img = load_image_rgb(p)
        if img is None: keep[i] = False; continue
        X[i] = cv2.resize(img, size).astype(np.float32) / 255.0
    return X[keep], m['quality_idx'].values[keep]
Xtr, ytr = load_split('train'); Xva, yva = load_split('val'); Xte, yte = load_split('test')
print("train", Xtr.shape, "| val", Xva.shape, "| test", Xte.shape)""")

M("## 2. Arquitectura")
C("""data_aug = models.Sequential([
    layers.RandomFlip("horizontal"), layers.RandomRotation(0.08),
    layers.RandomZoom(0.10), layers.RandomBrightness(0.10, value_range=(0,1)),
], name="augmentation")
def build_cnn():
    m = models.Sequential([
        layers.Input((*CNN_IMG_SIZE, 3)), data_aug,
        layers.Conv2D(32,3,padding='same',activation='relu'), layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.Conv2D(64,3,padding='same',activation='relu'), layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.Conv2D(128,3,padding='same',activation='relu'), layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(), layers.Dense(128,activation='relu'), layers.Dropout(0.5),
        layers.Dense(len(QUALITY_CLASSES), activation='softmax'),
    ], name="FruitCNN")
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m
cnn = build_cnn(); cnn.summary()""")

M("## 3. Entrenamiento")
C("""cw = compute_class_weight('balanced', classes=np.unique(ytr), y=ytr)
class_weight = dict(enumerate(cw))
print("class_weight:", {QUALITY_CLASSES[k]: round(v,2) for k,v in class_weight.items()})
cbs = [callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
       callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)]
history = cnn.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=40, batch_size=32,
                  class_weight=class_weight, callbacks=cbs, verbose=2)""")

M("## 4. Curvas de aprendizaje")
C("""h = history.history
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(h['loss'], label='train'); ax[0].plot(h['val_loss'], label='val'); ax[0].set_title('Pérdida'); ax[0].legend()
ax[1].plot(h['accuracy'], label='train'); ax[1].plot(h['val_accuracy'], label='val'); ax[1].set_title('Accuracy'); ax[1].legend()
plt.tight_layout(); plt.savefig(FIGURES_DIR / "fase4_cnn_curvas.pdf", bbox_inches="tight"); plt.show()""")

M("## 5. Evaluación en test")
C("""pred = cnn.predict(Xte, verbose=0).argmax(1)
acc = accuracy_score(yte, pred); f1 = f1_score(yte, pred, average='macro')
print(f"CNN -> accuracy={acc:.3f} | f1_macro={f1:.3f}\\n")
print(classification_report(yte, pred, target_names=QUALITY_CLASSES, digits=3))
fig, axx = plt.subplots(figsize=(5,4.5))
ConfusionMatrixDisplay(confusion_matrix(yte, pred), display_labels=QUALITY_CLASSES).plot(ax=axx, cmap='Greens', colorbar=False)
axx.set_title('CNN — Matriz de confusión (test)'); axx.tick_params(axis='x', rotation=30)
plt.tight_layout(); plt.savefig(FIGURES_DIR / "fase4_cnn_confusion.pdf", bbox_inches="tight"); plt.show()""")

M("## 6. Guardado")
C("""cnn.save(MODELS_DIR / "cnn_quality.keras")
pd.DataFrame({'modelo':['CNN'],'accuracy':[acc],'f1_macro':[f1]}).to_csv(MODELS_DIR / "cnn_metrics.csv", index=False)
print("Modelo guardado en models/saved/cnn_quality.keras")""")

M("➡️ **Siguiente:** Fase 5 — Evaluación comparativa.")

nb = new_notebook(cells=cells)
nb.metadata.kernelspec = {"display_name":"Python (fruit-quality)","language":"python","name":"fruit-quality"}
OUT.parent.mkdir(parents=True, exist_ok=True)
ExecutePreprocessor(timeout=3600, kernel_name="fruit-quality").preprocess(nb, {"metadata": {"path": str(OUT.parent)}})
nbf.write(nb, str(OUT))
print("OK ->", OUT)
