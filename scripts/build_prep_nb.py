"""Genera y ejecuta notebooks/3_data_preparation/preprocessing.ipynb."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks/3_data_preparation/preprocessing.ipynb"

cells = []
M = lambda s: cells.append(new_markdown_cell(s))
C = lambda s: cells.append(new_code_cell(s))

M("""# Fase 3 — Preparación de los Datos
**Proyecto:** FruitVision — Clasificación de Calidad de Frutas

---
Combina **dos fuentes** y produce un conjunto limpio, balanceado y dividido,
con **estimación de tamaño** por segmentación:

- **Kaggle** (`data/external`): GoodPremium, BadDescarte. La carpeta **Mixed se
  EXCLUYE** (varias frutas por foto, fondo no uniforme  se reserva para el
  ejercicio de segmentación).
- **Dataset propio** (`data/raw`): GoodPremium, **RegularEstándar**, BadDescarte.

La clase **Estándar** proviene ahora de imágenes propias *Regular* (1 fruta por
foto), reemplazando al antiguo "Mixed".

Pasos: (1) carga combinada, (2) **cap por fruta×calidad** (corrige sesgo
Pomegranate, EDA §2.3), (3) **split agrupado anti-fuga** 70/15/15, (4) tamaño,
(5) guardado.""")

C("""import sys, pathlib
ROOT = pathlib.Path.cwd()
while not (ROOT / "src").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style="whitegrid")

from src.config import (QUALITY_CLASSES, SIZE_CLASSES, CAP_PER_FRUIT_QUALITY,
                        FIGURES_DIR, PROCESSED_DIR)
from src.data import preprocessing as prep
from src.data.segmentation import segment_fruit
from src.data.paths import load_image_rgb
print("Repo:", ROOT)""")

M("## 1. Carga combinada (Kaggle sin Mixed + dataset propio)")
C("""raw = prep.load_combined_labels()
print("\\n[quality × source]")
print(pd.crosstab(raw['quality'], raw['source']).reindex(QUALITY_CLASSES))
print("\\n[fruta × calidad]")
print(pd.crosstab(raw['fruit'], raw['quality']).reindex(columns=QUALITY_CLASSES))""")

M("""## 2. Balanceo por *cap* (fruta × calidad)
El EDA (§2.3) detectó que **Pomegranate_Good** inflaba la clase Premium. Capeando
por *fruta × calidad* evitamos que una sola fruta domine una clase y reducimos el
desbalanceo. El residual lo absorbe `class_weight='balanced'`.""")
C("""capped = prep.apply_cap(raw)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
raw['quality'].value_counts().reindex(QUALITY_CLASSES).plot.bar(
    ax=ax[0], color="#c44", title=f"Antes del cap (n={len(raw)})")
capped['quality'].value_counts().reindex(QUALITY_CLASSES).plot.bar(
    ax=ax[1], color="#4a4", title=f"Después del cap={CAP_PER_FRUIT_QUALITY}/fruta×clase (n={len(capped)})")
for a in ax: a.set_xlabel("Calidad"); a.set_ylabel("Imágenes"); a.tick_params(axis='x', rotation=0)
plt.tight_layout(); plt.savefig(FIGURES_DIR / "fase3_balanceo.pdf", bbox_inches="tight"); plt.show()
print("Composición de Premium por fruta (antes vs después):")
print(pd.concat([
    raw[raw.quality=='Premium']['fruit'].value_counts(normalize=True).round(3).rename('antes'),
    capped[capped.quality=='Premium']['fruit'].value_counts(normalize=True).round(3).rename('después'),
], axis=1))""")

M("""## 3. Anti-fuga + split agrupado 70/15/15
El dataset trae **ráfagas de la misma fruta** (Kaggle y propio). Un split aleatorio
filtraría fotos casi idénticas entre train y test  **fuga de datos**. Lo evitamos:
agrupamos casi-duplicados con *perceptual hash* (dHash + Hamming ≤ 5)  `group_id`,
y hacemos el split **por grupo** dentro de cada clase.""")
C("""from src.data.dedup import assign_groups
capped = capped.copy()
capped['group_id'] = assign_groups(capped['abs_path'].tolist())
print(f"{len(capped)} imágenes -> {capped['group_id'].nunique()} grupos")

split_df = prep.grouped_split(capped)
print(pd.crosstab(split_df['quality'], split_df['split']).reindex(QUALITY_CLASSES)[['train','val','test']])
span = split_df.groupby('group_id')['split'].nunique()
print(f"\\nGrupos que cruzan >1 split (debe ser 0): {(span>1).sum()}")""")

M("""## 4. Estimación de tamaño por segmentación (C1)
Diámetro equivalente normalizado por la diagonal, discretizado en terciles
aprendidos **solo en train**. Ejecuta el pipeline completo y guarda los manifests.""")
C("""df = prep.build_manifests()
thr = df.attrs['size_thresholds']
print(f"\\nUmbrales de tamaño: Pequeño < {thr[0]:.3f} ≤ Mediano < {thr[1]:.3f} ≤ Grande")
print("\\n[quality × split]")
print(pd.crosstab(df['quality'], df['split']).reindex(QUALITY_CLASSES)[['train','val','test']])""")

M("### 4.1 Distribución de diámetro y clases de tamaño")
C("""fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(df['diameter_norm'].dropna(), bins=40, color="#69c", edgecolor="white")
ax[0].axvline(thr[0], color="red", ls="--", label=f"q33={thr[0]:.3f}")
ax[0].axvline(thr[1], color="darkred", ls="--", label=f"q66={thr[1]:.3f}")
ax[0].set_title("Diámetro normalizado"); ax[0].set_xlabel("diámetro / diagonal"); ax[0].legend()
pd.crosstab(df['size'], df['split']).reindex(SIZE_CLASSES)[['train','val','test']].plot.bar(ax=ax[1], stacked=True)
ax[1].set_title("Clase de tamaño por split"); ax[1].set_xlabel("Tamaño"); ax[1].tick_params(axis='x', rotation=0)
plt.tight_layout(); plt.savefig(FIGURES_DIR / "fase3_tamano.pdf", bbox_inches="tight"); plt.show()""")

M("### 4.2 Ejemplo visual de segmentación")
C("""import cv2
samples = df.sample(3, random_state=7)
fig, axes = plt.subplots(3, 2, figsize=(8, 11))
for i, (_, row) in enumerate(samples.iterrows()):
    img = load_image_rgb(row['abs_path']); seg = segment_fruit(img)
    x,y,w,h = seg.bbox; vis = img.copy()
    cv2.rectangle(vis, (x,y), (x+w,y+h), (255,0,0), max(2, img.shape[1]//150))
    axes[i,0].imshow(vis); axes[i,0].set_title(f"{row['fruit']} | {row['quality']} | {row['size']} | {row['source']}")
    axes[i,1].imshow(seg.mask, cmap='gray'); axes[i,1].set_title(f"máscara · diam_norm={seg.diameter_norm:.3f}")
    for a in axes[i]: a.axis('off')
plt.tight_layout(); plt.savefig(FIGURES_DIR / "fase3_segmentacion_ejemplos.pdf", bbox_inches="tight"); plt.show()""")

M("""## 4.3 Enriquecimiento con la carpeta Mixed (segmentación multi-fruta)
La carpeta Kaggle **Mixed** tiene varias frutas por foto. La **segmentamos en
recortes individuales** (cumple el requisito del enunciado) y los re-etiquetamos
por **daño superficial (heurística NTC-4580)**. Estos recortes se añaden **solo a
train** (nunca val/test)  enriquecen el entrenamiento sin meter etiquetas
derivadas de color en la evaluación (evita métricas circulares).""")
C("""print("Composición de TRAIN por fuente:")
tr = df[df['split']=='train']
print(pd.crosstab(tr['quality'], tr['source']).reindex(QUALITY_CLASSES))
# Ejemplos de recortes segmentados de Mixed
import cv2
seg = df[df['source']=='mixed_seg']
if len(seg):
    s = seg.sample(min(6, len(seg)), random_state=1)
    fig, axes = plt.subplots(1, len(s), figsize=(3*len(s), 3))
    if len(s)==1: axes=[axes]
    for ax,(_,r) in zip(axes, s.iterrows()):
        ax.imshow(load_image_rgb(r['abs_path'])); ax.axis('off')
        ax.set_title(f"{r['fruit']}\\n{r['quality']}", fontsize=9)
    plt.suptitle("Recortes individuales segmentados de Mixed (enriquecimiento train)")
    plt.tight_layout(); plt.savefig(FIGURES_DIR/"fase3_mixed_segmentado.pdf", bbox_inches="tight"); plt.show()""")

M("""## 5. Auditoría de label-noise (NTC-4580, umbral de daño por especie)
Aplicamos la heurística de daño a una muestra de **cada clase** para cuantificar:
- En **Premium/Descarte** (etiqueta de carpeta): qué fracción "re-etiquetaría" el
  daño  estimación de *label noise* y verificación del **piso de clase** (un
  Descarte no debería volverse Premium).
- En **Mixed**: cómo se reparte el daño en Premium/Estándar/Descarte.
Usa el umbral de daño **por especie** (FIX-PER-FRUIT: Banana 35, Pomegranate 40).""")
C("""from src.data.preprocessing import load_combined_labels
from src.data.segmentation import (segment_instances, compute_damage_pct,
                                    assign_quality_by_damage, FRUIT_DARK_V_THRESHOLD)
from src.config import EXTERNAL_DIR, MIXED_DIRNAME
print("Umbral 'oscuro' por especie:", FRUIT_DARK_V_THRESHOLD, "| resto=55")

cat = load_combined_labels()
def audit(sample_df, allow_multiple):
    recs=[]
    for _,r in sample_df.iterrows():
        im=load_image_rgb(r['abs_path']) if 'abs_path' in r else load_image_rgb(r['path'])
        if im is None: continue
        ins=segment_instances(im, allow_multiple=allow_multiple, fruit_name=r['fruit'])
        if not ins: continue
        for i in ins:
            recs.append({'fruit':r['fruit'],'origin':r['quality'],
                         'damage':i['damage_pct'],
                         'pred':assign_quality_by_damage(i['damage_pct'])})
    return pd.DataFrame(recs)

audit_rows=[]
for q in ['Premium','Descarte']:
    s=cat[cat['quality']==q].sample(min(300,len(cat[cat['quality']==q])),random_state=1)
    a=audit(s, allow_multiple=False); a['origin']=q; audit_rows.append(a)
# Mixed desde carpeta
import pandas as pd
mix=[{'fruit':p.parent.name.split('_')[0],'quality':'Mixed','path':str(p)}
     for p in (EXTERNAL_DIR/MIXED_DIRNAME).rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png'}]
mixdf=pd.DataFrame(mix).sample(min(300,len(mix)),random_state=1)
am=audit(mixdf, allow_multiple=True); am['origin']='Mixed'; audit_rows.append(am)
A=pd.concat(audit_rows, ignore_index=True)

print("\\n── Auditoría por clase de origen  etiqueta según daño ──")
for q in ['Premium','Descarte','Mixed']:
    sub=A[A['origin']==q]
    if not len(sub): continue
    dist=sub['pred'].value_counts(normalize=True).mul(100).round(1).to_dict()
    print(f"  {q:9s} (n={len(sub)}): {dist}")
    if q=='Premium':
        print(f"      label noise estimado (no-Premium): {100*(sub['pred']!='Premium').mean():.1f}%")
    if q=='Descarte':
        viol=100*(sub['pred']=='Premium').mean()
        print(f"      piso de clase: {viol:.1f}% ascendió a Premium ({'OK' if viol<2 else 'revisar'})")
fig,ax=plt.subplots(1,3,figsize=(15,4))
for a_,q in zip(ax,['Premium','Descarte','Mixed']):
    sub=A[A['origin']==q]['pred'].value_counts().reindex(QUALITY_CLASSES).fillna(0)
    sub.plot.bar(ax=a_, color=['#2ecc71','#f39c12','#e74c3c']); a_.set_title(f"Origen: {q}"); a_.tick_params(axis='x',rotation=0)
plt.suptitle("Auditoría de label-noise por clase (heurística de daño)")
plt.tight_layout(); plt.savefig(FIGURES_DIR/"fase3_label_noise_audit.pdf", bbox_inches="tight"); plt.show()""")

M("""## 6. Validaciones de integridad
Aserciones automáticas que garantizan un dataset sano antes de modelar.""")
C("""tr=df[df['split']=='train']; va=df[df['split']=='val']; te=df[df['split']=='test']

# 6.1 Anti-fuga: ningún grupo de casi-duplicados cruza particiones
span=df.groupby('group_id')['split'].nunique()
assert (span>1).sum()==0, "FUGA: hay grupos en más de un split"
print(" Anti-fuga: 0 grupos cruzan train/val/test")

# 6.2 Enriquecimiento Mixed solo en train (no en val/test)
assert (va['source']=='mixed_seg').sum()==0 and (te['source']=='mixed_seg').sum()==0, \\
    "Hay recortes mixed_seg en val/test"
print(" mixed_seg solo en train (val/test con etiquetas limpias)")

# 6.3 Sin NaN en columnas clave
for col in ['quality','quality_idx','fruit','source','size','split']:
    assert df[col].notna().all(), f"NaN en columna {col}"
print(" Sin valores nulos en columnas clave")

# 6.4 Todas las clases presentes en cada split
for nm,d_ in [('train',tr),('val',va),('test',te)]:
    assert set(d_['quality'].unique())>=set(QUALITY_CLASSES), f"Falta una clase en {nm}"
print(" Las 3 clases presentes en train/val/test")

# 6.5 Las imágenes existen en disco (muestra)
miss=sum(1 for p in df['abs_path'].sample(min(200,len(df)),random_state=0) if not __import__('pathlib').Path(p).exists())
assert miss==0, f"{miss} imágenes no existen en disco"
print(" Muestra de rutas verificada en disco")

print("\\nFuentes por split:")
for nm,d_ in [('train',tr),('val',va),('test',te)]:
    print(f"  {nm:5s} -> {d_['source'].value_counts().to_dict()}")
print("\\nProporción del split:", {k:round(v,3) for k,v in (df['split'].value_counts(normalize=True)).items()})""")

M("""## 7. Resumen de la Fase 3

**Entregables**

| Artefacto | Ubicación |
|---|---|
| Manifests train/val/test | `data/processed/manifest_*.csv` (col. `source`, `group_id`, `size`) |
| labels.csv limpio (un esquema) | `data/annotations/labels.csv` |
| Recortes Mixed segmentados | `data/processed/mixed_crops/<calidad>/<fruta>/` |
| Umbrales de tamaño | `data/processed/size_thresholds.json` |
| Figuras | `reports/figures/fase3_*.pdf` |

**Decisiones de diseño**

| Componente | Decisión |
|---|---|
| Fuentes | Kaggle (`data/external`) + propio (`data/raw`), escaneo de carpetas |
| Clase Estándar | Carpetas **Regular reales** de ambas fuentes (no Mixed) |
| Balanceo | Cap por **fruta×calidad** + `class_weight='balanced'` |
| Split | **Agrupado anti-fuga** (perceptual hash dHash) 70/15/15 estratificado |
| Mixed | **Segmentación multi-fruta + watershed** + re-etiquetado por daño  solo train |
| Daño | NTC-4580 con **umbral oscuro por especie** (FIX-PER-FRUIT) |
| Tamaño | Diámetro equivalente normalizado, terciles aprendidos en train |

**Limitaciones (para Fase 5)**
- La segmentación de Mixed separa frutas aisladas y muchas que se tocan (watershed),
  pero pilas muy abigarradas pueden agruparse (requeriría segmentación por instancias DL).
- Los umbrales de daño son heurísticos; los errores residuales se verán en la matriz
  de confusión de Fase 5.

 **Siguiente:** Fase 4 — Modelado.""")

nb = new_notebook(cells=cells)
nb.metadata.kernelspec = {"display_name":"Python (fruit-quality)","language":"python","name":"fruit-quality"}
OUT.parent.mkdir(parents=True, exist_ok=True)
ExecutePreprocessor(timeout=900, kernel_name="fruit-quality").preprocess(nb, {"metadata": {"path": str(OUT.parent)}})
nbf.write(nb, str(OUT))
print("OK ->", OUT)
