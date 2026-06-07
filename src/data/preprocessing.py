"""
C2/C3 — Preparación y balanceo de datos
========================================
Construye los manifests (train/val/test) a partir de DOS fuentes combinadas:

  - Kaggle (data/external): Good->Premium, Bad->Descarte.  (Mixed se EXCLUYE)
  - Propio (data/raw):      Good->Premium, Regular->Estándar, Bad->Descarte.

La clase **Estándar** proviene de las imágenes propias "Regular" (1 fruta por
foto), reemplazando a la carpeta Kaggle "Mixed" (varias frutas por foto).

Pasos:
  - Limpieza:  verificación de existencia de archivos.
  - Balanceo (C3):  cap por (fruta × calidad) para evitar que una fruta domine
                    una clase (hallazgo EDA §2.3) + class_weight en los modelos.
  - Split 70/15/15 POR GRUPO (anti-fuga) y estratificado por calidad.
  - Estimación de tamaño (C1) integrada: terciles aprendidos en train.

Salida: data/processed/manifest_{train,val,test}.csv con columnas
        [path, quality, quality_idx, fruit, source, size, diameter_norm,
         group_id, split].
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.data.dedup import assign_groups
from src.config import (
    CAP_MIXED_PER_FRUIT_QUALITY,
    CAP_PER_FRUIT_QUALITY,
    DAMAGE_PREMIUM_MAX,
    DAMAGE_STANDARD_MAX,
    EXTERNAL_DIR,
    FOLDER_QUALITY_MAP,
    INCLUDE_MIXED_ENRICHMENT,
    LABELS_CSV,
    MIXED_DIRNAME,
    MIXED_MIN_AREA_RATIO,
    PROCESSED_DIR,
    QUALITY_CLASSES,
    QUALITY_TO_IDX,
    RAW_DIR,
    SEED,
    TEST_RATIO,
    VAL_RATIO,
)
from src.data.paths import load_image_rgb, resolve_path
from src.data.segmentation import (
    assign_quality_by_damage,
    assign_size_class,
    compute_size_thresholds,
    segment_fruit,
    segment_instances,
)


_IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def _scan_folder_dataset(base_dir, source: str) -> pd.DataFrame:
    """Recorre carpetas '<Calidad> Quality_Fruits/<Fruta>_<Calidad>/*.jpg'.

    Mapea la calidad de la carpeta (Good/Regular/Bad) a la clase del proyecto
    según FOLDER_QUALITY_MAP. La fruta se toma del prefijo de la subcarpeta.
    Devuelve columnas [path, quality, fruit, source, abs_path].
    """
    rows = []
    if not base_dir.exists():
        return pd.DataFrame(columns=["path", "quality", "fruit", "source", "abs_path"])
    for quality_dir in sorted(base_dir.glob("* Quality_Fruits")):
        folder_quality = quality_dir.name.split(" ")[0]          # 'Good'/'Bad'/'Regular'
        cls = FOLDER_QUALITY_MAP.get(folder_quality)
        if cls is None:                                          # p. ej. 'Mixed' -> se ignora
            continue
        for fruit_dir in sorted(quality_dir.iterdir()):
            if not fruit_dir.is_dir():
                continue
            fruit = fruit_dir.name.split("_")[0]                 # 'Apple_Good' -> 'Apple'
            for img in fruit_dir.iterdir():
                if img.suffix.lower() in _IMG_EXTS:
                    rows.append({
                        "path": str(img.relative_to(base_dir.parents[1])),
                        "quality": cls, "fruit": fruit, "source": source,
                        "abs_path": img.resolve(),
                    })
    return pd.DataFrame(rows)


def load_combined_labels() -> pd.DataFrame:
    """Combina AMBAS fuentes escaneando carpetas (no depende de labels.csv).

    - Kaggle (`data/external`): Good→Premium, Regular→Estándar, Bad→Descarte.
    - Propio (`data/raw`):       Good→Premium, Regular→Estándar, Bad→Descarte.

    La carpeta Kaggle "Mixed Qualit_Fruits" NO coincide con el patrón
    "* Quality_Fruits", así que queda excluida automáticamente (se trata aparte
    como enriquecimiento). Escanear carpetas (en vez de leer labels.csv) hace que
    el Regular externo —clasificado a mano— entre sin depender de catálogos viejos.
    """
    kag = _scan_folder_dataset(EXTERNAL_DIR, "kaggle")
    own = _scan_folder_dataset(RAW_DIR, "propio")

    df = pd.concat([kag, own], ignore_index=True)
    exists = df["abs_path"].map(lambda p: p.exists())
    n_missing = int((~exists).sum())
    if n_missing:
        print(f"[clean] {n_missing} archivos no encontrados -> descartados")
    df = df[exists].reset_index(drop=True)
    print(f"[load] Kaggle={(df['source']=='kaggle').sum()} | "
          f"Propio={(df['source']=='propio').sum()} | Total={len(df)}")
    print(pd.crosstab(df["quality"], df["source"]).to_string())
    return df


def regenerate_labels_csv(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Regenera `data/annotations/labels.csv` con UN solo esquema, limpio.

    Esquema: [path, quality, quality_idx, fruit, source]. Refleja el catálogo
    combinado escaneado (3 clases, sin Mixed). Reemplaza el labels.csv antiguo
    que tenía esquema doble y rutas muertas (data/own inexistente).
    """
    if df is None:
        df = load_combined_labels()
    out = df[["path", "quality", "fruit", "source"]].copy()
    out["quality_idx"] = out["quality"].map(QUALITY_TO_IDX)
    out = out[["path", "quality", "quality_idx", "fruit", "source"]]
    out.to_csv(LABELS_CSV, index=False)
    print(f"[labels] {LABELS_CSV} regenerado: {len(out)} filas, esquema único")
    return out


def apply_cap(df: pd.DataFrame, cap: int = CAP_PER_FRUIT_QUALITY) -> pd.DataFrame:
    """Submuestrea cada combinación (fruta × calidad) a lo sumo a `cap` ejemplos.

    Evita que una fruta domine una clase (p. ej. Pomegranate_Good en Premium,
    hallazgo del EDA §2.3) y reduce el costo de cómputo.
    """
    parts = []
    for _, grp in df.groupby(["fruit", "quality"]):
        if len(grp) > cap:
            grp = grp.sample(cap, random_state=SEED)
        parts.append(grp)
    out = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return out


def stratified_split(df: pd.DataFrame) -> pd.DataFrame:
    """Asigna columna `split` estratificando por calidad (split simple por imagen).

    ⚠️ No usar si el dataset tiene casi-duplicados: provoca fuga de datos.
    Conservada para comparación; el pipeline usa `grouped_split`.
    """
    train_val, test = train_test_split(
        df, test_size=TEST_RATIO, stratify=df["quality"], random_state=SEED
    )
    val_size = VAL_RATIO / (1.0 - TEST_RATIO)
    train, val = train_test_split(
        train_val, test_size=val_size, stratify=train_val["quality"],
        random_state=SEED,
    )
    df = df.copy()
    df.loc[train.index, "split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"
    return df


def grouped_split(df: pd.DataFrame) -> pd.DataFrame:
    """Split 70/15/15 POR GRUPO (anti-fuga) y estratificado por calidad.

    Cada `group_id` (conjunto de casi-duplicados) se asigna **entero** a una sola
    partición. La asignación se hace a nivel de GRUPO, estratificando por la clase
    representativa del grupo (su moda). Esto es robusto incluso si un grupo de
    casi-duplicados abarca imágenes de distinta clase (caso raro pero posible en el
    dataset combinado): el grupo completo va a un único split → cero fuga.
    """
    import numpy as np
    df = df.copy()
    df["split"] = None

    # Clase representativa y un orden reproducible por grupo.
    grp_quality = df.groupby("group_id")["quality"].agg(lambda s: s.mode().iat[0])
    rng = np.random.RandomState(SEED)

    for cls in df["quality"].unique():
        gids = grp_quality[grp_quality == cls].index.to_numpy()
        rng.shuffle(gids)
        n = len(gids)
        n_test = int(round(n * TEST_RATIO))
        n_val = int(round(n * VAL_RATIO))
        test_g = set(gids[:n_test])
        val_g = set(gids[n_test:n_test + n_val])
        train_g = set(gids[n_test + n_val:])
        for split, gset in (("train", train_g), ("val", val_g), ("test", test_g)):
            df.loc[df["group_id"].isin(gset), "split"] = split
    return df


def estimate_sizes(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula diámetro normalizado por imagen y asigna clase de tamaño.

    Los umbrales (terciles) se aprenden SOLO con el split de entrenamiento para
    evitar fuga de información, y se aplican a val/test.
    """
    from tqdm.auto import tqdm

    diam = []
    for p in tqdm(df["abs_path"], desc="segmentando"):
        img = load_image_rgb(p)
        diam.append(segment_fruit(img).diameter_norm if img is not None else float("nan"))
    df = df.copy()
    df["diameter_norm"] = diam

    train_diam = df.loc[df["split"] == "train", "diameter_norm"].dropna().values
    thresholds = compute_size_thresholds(train_diam)
    df["size"] = df["diameter_norm"].map(
        lambda d: assign_size_class(d, thresholds) if pd.notna(d) else "Mediano"
    )
    df.attrs["size_thresholds"] = thresholds
    return df


def build_mixed_enrichment(thresholds: tuple[float, float]) -> pd.DataFrame:
    """Segmenta la carpeta Mixed en frutas individuales y las re-etiqueta por daño.

    Cada recorte se guarda en disco y se devuelve como fila lista para añadir a
    TRAIN (nunca val/test). Cumple el requisito del enunciado de segmentar Mixed
    y enriquece el dataset sin meter etiquetas derivadas de color en la evaluación.
    """
    import cv2
    from tqdm.auto import tqdm

    mixed_dir = EXTERNAL_DIR / MIXED_DIRNAME
    if not mixed_dir.exists():
        print(f"[mixed] carpeta no encontrada: {mixed_dir} -> sin enriquecimiento")
        return pd.DataFrame()

    crops_root = PROCESSED_DIR / "mixed_crops"
    root = PROCESSED_DIR.parents[1]                  # raíz del repo (para rutas relativas)
    rows = []
    img_paths = [p for p in mixed_dir.rglob("*")
                 if p.suffix.lower() in _IMG_EXTS and p.is_file()]
    for p in tqdm(img_paths, desc="mixed-seg"):
        fruit = p.parent.name.split("_")[0]
        img = load_image_rgb(p)
        if img is None:
            continue
        for k, inst in enumerate(segment_instances(img, allow_multiple=True,
                                                   min_area_ratio=MIXED_MIN_AREA_RATIO,
                                                   fruit_name=fruit)):
            quality = assign_quality_by_damage(inst["damage_pct"],
                                               DAMAGE_PREMIUM_MAX, DAMAGE_STANDARD_MAX)
            out_dir = crops_root / quality / fruit
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{p.stem}__seg{k:02d}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(inst["crop_rgb"], cv2.COLOR_RGB2BGR))
            rows.append({
                "path": str(out_path.relative_to(root)),
                "quality": quality, "fruit": fruit, "source": "mixed_seg",
                "diameter_norm": inst["diameter_norm"],
                "size": assign_size_class(inst["diameter_norm"], thresholds),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Cap por fruta×calidad para que el enriquecimiento no domine ninguna clase.
    parts = [g.sample(min(len(g), CAP_MIXED_PER_FRUIT_QUALITY), random_state=SEED)
             for _, g in df.groupby(["fruit", "quality"])]
    df = pd.concat(parts).reset_index(drop=True)
    df["split"] = "train"
    df["quality_idx"] = df["quality"].map(QUALITY_TO_IDX)
    print(f"[mixed] {len(df)} recortes de enriquecimiento añadidos a train")
    print(df["quality"].value_counts().to_dict())
    return df


def build_manifests(cap: int = CAP_PER_FRUIT_QUALITY, with_size: bool = True) -> pd.DataFrame:
    """Pipeline completo. Devuelve el DataFrame y guarda los manifests."""
    df = load_combined_labels()
    regenerate_labels_csv(df)          # labels.csv limpio (un solo esquema)
    df = apply_cap(df, cap)
    # Anti-fuga: agrupar casi-duplicados y dividir por grupo.
    df["group_id"] = assign_groups(df["abs_path"].tolist())
    df = grouped_split(df)
    thresholds = (float("nan"), float("nan"))
    if with_size:
        df = estimate_sizes(df)
        # Capturar los umbrales explícitamente (df.attrs no sobrevive a concat).
        thresholds = compute_size_thresholds(
            df.loc[df["split"] == "train", "diameter_norm"].dropna().values)
    else:
        df["diameter_norm"] = float("nan")
        df["size"] = "Mediano"

    df["quality_idx"] = df["quality"].map(QUALITY_TO_IDX)

    # Enriquecimiento Mixed (segmentado) -> SOLO train.
    if with_size and INCLUDE_MIXED_ENRICHMENT:
        mixed = build_mixed_enrichment(thresholds)
        if not mixed.empty:
            mixed["group_id"] = range(int(df["group_id"].max()) + 1,
                                      int(df["group_id"].max()) + 1 + len(mixed))
            mixed["abs_path"] = mixed["path"].map(resolve_path)
            df = pd.concat([df, mixed], ignore_index=True)

    df.attrs["size_thresholds"] = thresholds

    # Guardar un manifest por split con la ruta relativa original (portable).
    cols = ["path", "quality", "quality_idx", "fruit", "source", "size",
            "diameter_norm", "group_id", "split"]
    for split in ("train", "val", "test"):
        sub = df.loc[df["split"] == split, cols]
        sub.to_csv(PROCESSED_DIR / f"manifest_{split}.csv", index=False)
    # Persistir los umbrales de tamaño para reutilizarlos con imágenes nuevas.
    if with_size:
        import json
        (PROCESSED_DIR / "size_thresholds.json").write_text(
            json.dumps({"q33": thresholds[0], "q66": thresholds[1]}))
    # Invalidar el cache de features: los manifests cambiaron y las features
    # cacheadas (.npy) quedarían desalineadas con las nuevas particiones.
    for f in list(PROCESSED_DIR.glob("X_*.npy")) + list(PROCESSED_DIR.glob("y_*.npy")):
        f.unlink()
    print(f"[build] manifests guardados en {PROCESSED_DIR} (cache de features invalidado)")
    return df


def load_manifest(split: str) -> pd.DataFrame:
    """Carga un manifest y añade la columna `abs_path` resuelta."""
    df = pd.read_csv(PROCESSED_DIR / f"manifest_{split}.csv")
    df["abs_path"] = df["path"].map(resolve_path)
    return df


def load_size_thresholds() -> tuple[float, float]:
    """Lee los umbrales de tamaño aprendidos en train (build_manifests)."""
    import json
    d = json.loads((PROCESSED_DIR / "size_thresholds.json").read_text())
    return d["q33"], d["q66"]


def build_own_manifest(raw_dir=None) -> pd.DataFrame:
    """Construye el manifest OOD a partir de `data/annotations/labels_own.csv`.

    Para CADA imagen propia:
      - resuelve su ruta en `data/raw/` (o `raw_dir`),
      - estima el tamaño con los MISMOS umbrales aprendidos en train,
      - mapea la calidad anotada a su índice.
    Las imágenes propias se usan SOLO para evaluación (test OOD), nunca para
    entrenar, así que no se tocan los modelos.
    """
    from src.config import ANNOTATIONS_DIR, DATA_DIR
    raw_dir = DATA_DIR / "raw" if raw_dir is None else raw_dir

    own = pd.read_csv(ANNOTATIONS_DIR / "labels_own.csv")
    own = own[own["quality"].isin(QUALITY_CLASSES)].copy()
    own["abs_path"] = own["filename"].map(lambda f: (raw_dir / f).resolve())
    exists = own["abs_path"].map(lambda p: p.exists())
    if (~exists).any():
        print(f"[own] {int((~exists).sum())} archivos no encontrados en {raw_dir}")
    own = own[exists].reset_index(drop=True)

    thr = load_size_thresholds()
    diam, size = [], []
    for p in own["abs_path"]:
        img = load_image_rgb(p)
        d = segment_fruit(img).diameter_norm if img is not None else float("nan")
        diam.append(d)
        size.append(assign_size_class(d, thr) if pd.notna(d) else "Mediano")
    own["diameter_norm"] = diam
    own["size_pred"] = size                      # tamaño estimado por segmentación
    own["quality_idx"] = own["quality"].map(QUALITY_TO_IDX)
    own["path"] = own["abs_path"].map(str)
    own.to_csv(PROCESSED_DIR / "manifest_own.csv", index=False)
    print(f"[own] manifest_own.csv con {len(own)} imágenes -> {PROCESSED_DIR}")
    return own
