"""
C2/C3 — Preparación y balanceo de datos
========================================
Construye los manifests (train/val/test) a partir del `labels.csv`, aplicando:

  - Limpieza:  verificación de existencia de archivos.
  - Balanceo (C3):  cap por clase de calidad para mitigar el desbalanceo 10:1.
                    El residual lo absorben los modelos con class_weight.
  - Split estratificado 70/15/15 por calidad (reproducible con SEED).
  - Estimación de tamaño (C1) integrada: terciles aprendidos en train.

Salida: data/processed/manifest_{train,val,test}.csv con columnas
        [path, quality, quality_idx, fruit, size, diameter_norm, split].
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.data.dedup import assign_groups
from src.config import (
    CAP_PER_QUALITY,
    LABELS_CSV,
    PROCESSED_DIR,
    QUALITY_CLASSES,
    QUALITY_TO_IDX,
    SEED,
    TEST_RATIO,
    VAL_RATIO,
)
from src.data.paths import load_image_rgb, resolve_path
from src.data.segmentation import (
    assign_size_class,
    compute_size_thresholds,
    segment_fruit,
)


def load_clean_labels() -> pd.DataFrame:
    """Carga labels.csv y elimina filas cuyo archivo no existe en disco."""
    df = pd.read_csv(LABELS_CSV)
    df = df[df["quality"].isin(QUALITY_CLASSES)].copy()
    df["abs_path"] = df["path"].map(lambda p: resolve_path(p))
    exists = df["abs_path"].map(lambda p: p.exists())
    n_missing = int((~exists).sum())
    if n_missing:
        print(f"[clean] {n_missing} archivos no encontrados -> descartados")
    return df[exists].reset_index(drop=True)


def apply_cap(df: pd.DataFrame, cap: int = CAP_PER_QUALITY) -> pd.DataFrame:
    """Submuestrea cada clase de calidad a lo sumo a `cap` ejemplos.

    Reduce el desbalanceo (Premium 11664 -> cap) y el costo de cómputo, sin
    tocar la clase minoritaria (Estándar) que se conserva completa.
    """
    parts = []
    for cls, grp in df.groupby("quality"):
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

    Todas las fotos de una misma fruta (`group_id`) caen en la misma partición.
    Como cada grupo de casi-duplicados pertenece a una sola clase de calidad,
    aplicamos GroupShuffleSplit **dentro de cada clase** -> se preserva a la vez
    la integridad de grupos y la proporción de clases.
    """
    df = df.copy()
    df["split"] = None
    val_size = VAL_RATIO / (1.0 - TEST_RATIO)

    for _, cls_df in df.groupby("quality"):
        idx = cls_df.index.values
        groups = cls_df["group_id"].values

        gss_test = GroupShuffleSplit(n_splits=1, test_size=TEST_RATIO, random_state=SEED)
        trval_loc, test_loc = next(gss_test.split(idx, groups=groups))
        trval_idx, test_idx = idx[trval_loc], idx[test_loc]

        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=SEED)
        tr_loc, val_loc = next(gss_val.split(trval_idx, groups=groups[trval_loc]))
        train_idx, val_idx = trval_idx[tr_loc], trval_idx[val_loc]

        df.loc[train_idx, "split"] = "train"
        df.loc[val_idx, "split"] = "val"
        df.loc[test_idx, "split"] = "test"
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


def build_manifests(cap: int = CAP_PER_QUALITY, with_size: bool = True) -> pd.DataFrame:
    """Pipeline completo. Devuelve el DataFrame y guarda los manifests."""
    df = load_clean_labels()
    df = apply_cap(df, cap)
    # Anti-fuga: agrupar casi-duplicados y dividir por grupo.
    df["group_id"] = assign_groups(df["abs_path"].tolist())
    df = grouped_split(df)
    if with_size:
        df = estimate_sizes(df)
    else:
        df["diameter_norm"] = float("nan")
        df["size"] = "Mediano"

    df["quality_idx"] = df["quality"].map(QUALITY_TO_IDX)

    # Guardar un manifest por split con la ruta relativa original (portable).
    cols = ["path", "quality", "quality_idx", "fruit", "size", "diameter_norm",
            "group_id", "split"]
    for split in ("train", "val", "test"):
        sub = df.loc[df["split"] == split, cols]
        sub.to_csv(PROCESSED_DIR / f"manifest_{split}.csv", index=False)
    # Persistir los umbrales de tamaño para reutilizarlos con imágenes nuevas.
    if with_size:
        import json
        (PROCESSED_DIR / "size_thresholds.json").write_text(
            json.dumps({"q33": df.attrs["size_thresholds"][0],
                        "q66": df.attrs["size_thresholds"][1]})
        )
    print(f"[build] manifests guardados en {PROCESSED_DIR}")
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
