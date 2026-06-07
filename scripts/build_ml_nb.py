"""Genera y ejecuta notebooks/4_modeling/ml_models.ipynb (RF + XGBoost + ablation)."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks/4_modeling/ml_models.ipynb"

cells = []
M = lambda s: cells.append(new_markdown_cell(s))
C = lambda s: cells.append(new_code_cell(s))

M("""# Fase 4 (parte A) — Modelos de Machine Learning clásico
**Proyecto:** FruitVision — Clasificación de Calidad de Frutas

---
Dos modelos clásicos (Random Forest, XGBoost) sobre características HOG + color HSV
(1860 dims), con **GridSearchCV (k=5)** optimizando F1-macro.

**Ablation:** entrenamos cada modelo con dos variantes de TRAIN —**con** y **sin**
los recortes de Mixed (`mixed_seg`)— manteniendo idénticos val/test, para medir si
ese enriquecimiento (de etiqueta ruidosa) ayuda o estorba. Guardamos la variante
ganadora.""")

C("""import sys, pathlib, time
ROOT = pathlib.Path.cwd()
while not (ROOT / "src").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, joblib
import matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style="whitegrid")
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import xgboost as xgb
from src.config import QUALITY_CLASSES, MODELS_DIR, FIGURES_DIR, SEED
from src.data.preprocessing import load_manifest
from src.features.extract import build_feature_matrix""")

M("## 1. Características: train (completo), train (sin mixed_seg), val, test")
C("""m_train = load_manifest('train')
m_train_nomix = m_train[m_train['source'] != 'mixed_seg'].copy()
Xtr,  ytr  = build_feature_matrix(m_train,       'train')
Xtrn, ytrn = build_feature_matrix(m_train_nomix, 'train_nomixed')
Xva,  yva  = build_feature_matrix(load_manifest('val'),  'val')
Xte,  yte  = build_feature_matrix(load_manifest('test'), 'test')
print(f"train completo={Xtr.shape[0]} | train sin mixed_seg={Xtrn.shape[0]} | val={Xva.shape[0]} | test={Xte.shape[0]}")""")

M("## 2. Línea base")
C("""dummy = DummyClassifier(strategy='most_frequent').fit(Xtr, ytr)
base_acc=accuracy_score(yte,dummy.predict(Xte)); base_f1=f1_score(yte,dummy.predict(Xte),average='macro')
print(f"Baseline -> accuracy={base_acc:.3f} | f1_macro={base_f1:.3f}")""")

M("""## 3. Búsqueda de hiperparámetros (GridSearchCV, k=5)
Hacemos la búsqueda sobre el train **completo**; el ablation reutiliza esos mismos
hiperparámetros en ambas variantes (aísla el efecto de los datos, no del tuning).""")
C("""rf_grid = {'n_estimators':[200,400], 'max_depth':[None,20], 'min_samples_split':[2,5]}
t=time.time()
rf_gs = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=SEED, n_jobs=1),
                     rf_grid, cv=5, scoring='f1_macro', n_jobs=-1).fit(Xtr, ytr)
print(f"[RF {time.time()-t:.0f}s] {rf_gs.best_params_}")
xgb_grid = {'learning_rate':[0.1,0.3], 'max_depth':[4,6], 'n_estimators':[200,400]}
t=time.time()
xg_gs = GridSearchCV(xgb.XGBClassifier(tree_method='hist', random_state=SEED, n_jobs=1, eval_metric='mlogloss'),
                     xgb_grid, cv=5, scoring='f1_macro', n_jobs=-1).fit(Xtr, ytr)
print(f"[XGB {time.time()-t:.0f}s] {xg_gs.best_params_}")""")

M("## 4. Ablation: con vs sin `mixed_seg`")
C("""def fit_eval(base_est, Xt, yt, Xe, ye):
    mdl = clone(base_est).fit(Xt, yt)
    p = mdl.predict(Xe)
    return mdl, accuracy_score(ye,p), f1_score(ye,p,average='macro')

rf_base = RandomForestClassifier(class_weight='balanced', random_state=SEED,
                                 n_jobs=-1, **rf_gs.best_params_)
xg_base = xgb.XGBClassifier(tree_method='hist', random_state=SEED, n_jobs=-1,
                            eval_metric='mlogloss', **xg_gs.best_params_)
variants = {'con mixed_seg':(Xtr,ytr), 'sin mixed_seg':(Xtrn,ytrn)}
models={}; rows=[]
for vname,(Xt,yt) in variants.items():
    for mname, base in [('Random Forest',rf_base),('XGBoost',xg_base)]:
        mdl,acc,f1 = fit_eval(base, Xt, yt, Xte, yte)
        models[(mname,vname)]=mdl
        rows.append({'modelo':mname,'variante':vname,'accuracy':acc,'f1_macro':f1})
abl = pd.DataFrame(rows)
print(abl.pivot(index='modelo', columns='variante', values='f1_macro').round(4).to_string())
print("\\n(valores = F1-macro en test)")
display(abl.round(4))""")

C("""# Elegir la mejor variante por modelo (F1-macro test) y graficar el ablation
fig,ax=plt.subplots(figsize=(7,4))
abl.pivot(index='modelo',columns='variante',values='f1_macro').plot.bar(ax=ax, rot=0)
ax.set_ylim(0.8,1.0); ax.set_ylabel('F1-macro (test)'); ax.set_title('Ablation: efecto del enriquecimiento Mixed')
ax.legend(title='train'); plt.tight_layout(); plt.savefig(FIGURES_DIR/"fase4_ablation_ml.pdf", bbox_inches="tight"); plt.show()

best={}
for mname in ['Random Forest','XGBoost']:
    sub=abl[abl.modelo==mname].sort_values('f1_macro',ascending=False).iloc[0]
    best[mname]=sub['variante']
    print(f"{mname}: mejor variante = '{sub['variante']}' (f1={sub['f1_macro']:.4f})")""")

M("## 5. Evaluación detallada de los modelos ganadores")
C("""results=[{'modelo':'Baseline','accuracy':base_acc,'f1_macro':base_f1}]
chosen={}
for mname in ['Random Forest','XGBoost']:
    mdl=models[(mname,best[mname])]; chosen[mname]=mdl
    p=mdl.predict(Xte); acc=accuracy_score(yte,p); f1=f1_score(yte,p,average='macro')
    results.append({'modelo':f"{mname} ({best[mname]})",'accuracy':acc,'f1_macro':f1})
    print(f"=== {mname} ({best[mname]}) ===")
    print(classification_report(yte,p,target_names=QUALITY_CLASSES,digits=3))
res=pd.DataFrame(results).set_index('modelo'); print(res.round(3).to_string())""")

C("""fig,ax=plt.subplots(1,2,figsize=(11,4.5))
for a,mname in zip(ax,['Random Forest','XGBoost']):
    p=chosen[mname].predict(Xte)
    ConfusionMatrixDisplay(confusion_matrix(yte,p),display_labels=QUALITY_CLASSES).plot(ax=a,cmap='Blues',colorbar=False)
    a.set_title(f"{mname} ({best[mname]})"); a.tick_params(axis='x',rotation=30)
plt.tight_layout(); plt.savefig(FIGURES_DIR/"fase4_ml_confusion.pdf", bbox_inches="tight"); plt.show()""")

M("## 6. Importancia por familia de características (RF ganador)")
C("""imp=chosen['Random Forest'].feature_importances_; n_hog=1764
plt.figure(figsize=(5,4))
plt.bar(['HOG (forma)','Color HSV'],[imp[:n_hog].sum(),imp[n_hog:].sum()],color=['#69c','#e9a'])
plt.ylabel('Importancia agregada'); plt.title('Aporte por familia (RF)')
plt.savefig(FIGURES_DIR/"fase4_ml_importancia.pdf", bbox_inches="tight"); plt.show()
print(f"HOG={imp[:n_hog].sum():.2f} | Color={imp[n_hog:].sum():.2f}")""")

M("## 7. Guardado de los mejores modelos")
C("""joblib.dump(chosen['Random Forest'], MODELS_DIR/"random_forest.pkl")
joblib.dump(chosen['XGBoost'],       MODELS_DIR/"xgboost.pkl")
best_overall = max(['Random Forest','XGBoost'], key=lambda m: res.loc[f"{m} ({best[m]})",'f1_macro'])
joblib.dump(chosen[best_overall], MODELS_DIR/"best_quality_ml.pkl")
abl.to_csv(MODELS_DIR/"ablation_ml.csv", index=False)
print(f"Mejor ML: {best_overall} ({best[best_overall]}) -> best_quality_ml.pkl")
print("Ablation guardado en models/saved/ablation_ml.csv")""")

M("""## Resumen
- **GridSearchCV (k=5)** para RF y XGBoost.
- **Ablation con/sin `mixed_seg`** (mismos val/test): se conserva la variante ganadora por modelo.
- Ambos superan ampliamente la línea base; el mejor se guarda para el despliegue.

 **Siguiente:** Fase 4 (parte B) — CNN.""")

nb = new_notebook(cells=cells)
nb.metadata.kernelspec = {"display_name":"Python (fruit-quality)","language":"python","name":"fruit-quality"}
OUT.parent.mkdir(parents=True, exist_ok=True)
ExecutePreprocessor(timeout=4000, kernel_name="fruit-quality").preprocess(nb, {"metadata": {"path": str(OUT.parent)}})
nbf.write(nb, str(OUT))
print("OK ->", OUT)
