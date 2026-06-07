"""Genera y ejecuta notebooks/5_evaluation/model_evaluation.ipynb."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks/5_evaluation/model_evaluation.ipynb"

cells = []
M = lambda s: cells.append(new_markdown_cell(s))
C = lambda s: cells.append(new_code_cell(s))

M("""# Fase 5 — Evaluación comparativa unificada
**Proyecto:** FruitVision — Clasificación de Calidad de Frutas

---
Compara en igualdad de condiciones (mismo test, split **sin fuga**) los modelos
ya entrenados: **Baseline, Random Forest, XGBoost, CNN**. Incluye tabla,
matrices de confusión, **análisis de errores** y un **desglose por fuente**
(Kaggle vs propio) que mide la generalización al dataset recolectado por el grupo.

Este notebook **NO entrena**: carga los modelos guardados y los evalúa.""")

C("""import sys, pathlib, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
ROOT = pathlib.Path.cwd()
while not (ROOT / "src").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, joblib, cv2, collections
import matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style="whitegrid")
import tensorflow as tf
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from src.config import QUALITY_CLASSES, CNN_IMG_SIZE, MODELS_DIR, FIGURES_DIR
from src.data.preprocessing import load_manifest
from src.features.extract import build_feature_matrix
from src.data.paths import load_image_rgb""")

M("## 1. Cargar modelos y datos de prueba")
C("""Xtr, ytr = build_feature_matrix(load_manifest('train'), 'train')
te = load_manifest('test')
Xte, yte = build_feature_matrix(te, 'test')
def load_imgs(manifest, size=CNN_IMG_SIZE):
    X = np.zeros((len(manifest), *size, 3), np.float32)
    for i, p in enumerate(manifest['abs_path']):
        img = load_image_rgb(p)
        if img is not None: X[i] = cv2.resize(img, size).astype(np.float32) / 255.0
    return X
Xte_img = load_imgs(te)
rf = joblib.load(MODELS_DIR/"random_forest.pkl"); xgbm = joblib.load(MODELS_DIR/"xgboost.pkl")
cnn = tf.keras.models.load_model(MODELS_DIR/"cnn_quality.keras")
dummy = DummyClassifier(strategy='most_frequent').fit(Xtr, ytr)
print("Test:", len(te), "| por fuente:", te['source'].value_counts().to_dict())""")

M("## 2. Predicciones")
C("""preds = {'Baseline': dummy.predict(Xte), 'Random Forest': rf.predict(Xte),
         'XGBoost': xgbm.predict(Xte), 'CNN': cnn.predict(Xte_img, verbose=0).argmax(1)}""")

M("## 3. Tabla comparativa")
C("""results = pd.DataFrame([{'modelo':n, 'accuracy':accuracy_score(yte,p),
    'f1_macro':f1_score(yte,p,average='macro')} for n,p in preds.items()]).set_index('modelo')
print(results.round(3))
ax = results.plot.bar(figsize=(8,4.5), rot=0); ax.set_ylim(0,1); ax.legend(loc='lower right')
ax.set_title("Comparativa de modelos (test, split sin fuga)"); ax.set_ylabel("score")
plt.tight_layout(); plt.savefig(FIGURES_DIR/"fase5_comparativa.pdf", bbox_inches="tight"); plt.show()""")

M("## 4. Métricas por clase y matrices de confusión")
C("""for name in ['Random Forest','XGBoost','CNN']:
    print(f"=== {name} ==="); print(classification_report(yte, preds[name], target_names=QUALITY_CLASSES, digits=3))""")
C("""fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for a, name in zip(ax, ['Random Forest','XGBoost','CNN']):
    ConfusionMatrixDisplay(confusion_matrix(yte, preds[name]), display_labels=QUALITY_CLASSES).plot(ax=a, cmap='Blues', colorbar=False)
    a.set_title(name); a.tick_params(axis='x', rotation=30)
plt.tight_layout(); plt.savefig(FIGURES_DIR/"fase5_confusion.pdf", bbox_inches="tight"); plt.show()""")

M("## 5. Análisis de errores (mejor modelo)")
C("""best_name = results.drop('Baseline')['f1_macro'].idxmax(); best = preds[best_name]
wrong = np.where(best != yte)[0]
print(f"Mejor modelo: {best_name} | errores: {len(wrong)}/{len(yte)}")
pairs = collections.Counter((QUALITY_CLASSES[yte[i]], QUALITY_CLASSES[best[i]]) for i in wrong)
print("\\nConfusiones (real -> predicho):")
for (a,b),c in pairs.most_common(5): print(f"  {a:9s} -> {b:9s}: {c}")
sample = wrong[:6]
fig, axes = plt.subplots(2, 3, figsize=(11, 7))
for ax, i in zip(axes.flatten(), sample):
    ax.imshow(load_image_rgb(te.iloc[i]['abs_path'])); ax.axis('off')
    ax.set_title(f"real: {QUALITY_CLASSES[yte[i]]} | pred: {QUALITY_CLASSES[best[i]]}", fontsize=9)
for ax in axes.flatten()[len(sample):]: ax.axis('off')
plt.suptitle(f"Errores de {best_name}"); plt.tight_layout()
plt.savefig(FIGURES_DIR/"fase5_errores.pdf", bbox_inches="tight"); plt.show()""")

M("""## 6. Desglose por fuente: Kaggle vs Dataset propio
Medimos el desempeño por separado en imágenes de Kaggle y en las recolectadas por
el grupo. Una caída en el dataset propio indica *domain shift* (otra cámara/luz/
fondo) — la prueba de generalización real. La clase **Estándar** solo aparece en
las imágenes propias (es la fuente de esa clase).""")
C("""rows = []
for src in ['kaggle','propio']:
    mask = (te['source']==src).values
    if mask.sum()==0: continue
    for name,p in preds.items():
        if name=='Baseline': continue
        rows.append({'fuente':src, 'modelo':name, 'n':int(mask.sum()),
                     'accuracy':accuracy_score(yte[mask], p[mask]),
                     'f1_macro':f1_score(yte[mask], p[mask], average='macro')})
by_source = pd.DataFrame(rows).set_index(['fuente','modelo'])
print(by_source.round(3))""")

M("""## 7. Sobre la salida de **tamaño** (aclaración importante)
El tamaño NO lo predice un modelo: se mide por segmentación como
**diámetro equivalente / diagonal de la imagen**. Es decir, refleja **cuánto llena
la fruta el encuadre**, no su diámetro físico real (que requeriría una referencia
de escala en la foto). Por eso aparece correlacionado con la fuente/tipo de foto.""")
C("""import pandas as pd
ct = pd.crosstab(te['quality'], te['size']).reindex(index=QUALITY_CLASSES,
        columns=['Pequeño','Mediano','Grande'])
print("Tamaño (estimado) × calidad en test:"); print(ct.to_string())
print("\\n→ El tamaño es 'fracción del encuadre ocupada por la fruta', medida")
print("  reproducible y útil para la app, pero NO equivale a tamaño físico.")""")

M("## 8. Ablation (resumen): ¿ayudó el enriquecimiento Mixed?")
C("""import pandas as pd
from src.config import MODELS_DIR
for name,f in [('ML (RF/XGB)','ablation_ml.csv'),('CNN','ablation_cnn.csv')]:
    p=MODELS_DIR/f
    if p.exists():
        print(f"=== {name} ===)"); print(pd.read_csv(p).round(4).to_string(index=False)); print()
print("Los modelos guardados corresponden a la variante ganadora por modelo.")""")

M("""## 9. Resumen de la Fase 5
- Comparación justa (mismo test, split **sin fuga**), 3 clases balanceadas en test.
- Tabla, matrices de confusión y **análisis de errores**.
- **Desglose por fuente** (Kaggle vs propio) → evidencia de generalización.
- **Ablation** con/sin enriquecimiento Mixed: se conserva la variante ganadora.
- **Tamaño**: medición por segmentación (fracción del encuadre), no físico.""")

nb = new_notebook(cells=cells)
nb.metadata.kernelspec = {"display_name":"Python (fruit-quality)","language":"python","name":"fruit-quality"}
OUT.parent.mkdir(parents=True, exist_ok=True)
ExecutePreprocessor(timeout=1200, kernel_name="fruit-quality").preprocess(nb, {"metadata": {"path": str(OUT.parent)}})
nbf.write(nb, str(OUT))
print("OK ->", OUT)
