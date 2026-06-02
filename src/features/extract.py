"""
C4 — Extracción de características para modelos ML clásicos
==========================================================
Los modelos clásicos (Random Forest, XGBoost) no operan sobre píxeles crudos;
necesitan un vector de características descriptivo. Combinamos dos familias
complementarias:

  - HOG (Histogram of Oriented Gradients): captura FORMA y bordes (textura,
    contorno de la fruta, presencia de golpes/hendiduras).
  - Histograma de color en HSV: captura APARIENCIA cromática (madurez, manchas,
    pardeamiento), invariante a la posición del píxel.

El vector final es la concatenación de ambos. Se cachea en disco (.npy) porque
extraer HOG de miles de imágenes es costoso y no queremos repetirlo.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog
from tqdm.auto import tqdm

from src.config import FEATURE_IMG_SIZE, PROCESSED_DIR
from src.data.paths import load_image_rgb

# Parámetros HOG (fijos para reproducibilidad).
_HOG_KW = dict(
    orientations=9,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    transform_sqrt=True,
)
_COLOR_BINS = 32  # bins por canal HSV


def extract_features(image_rgb: np.ndarray) -> np.ndarray:
    """Devuelve el vector de características (HOG + histograma de color HSV)."""
    img = cv2.resize(image_rgb, FEATURE_IMG_SIZE)

    # --- HOG sobre escala de grises (forma/bordes) ---
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hog_vec = hog(gray, **_HOG_KW)

    # --- Histograma de color en HSV, normalizado (apariencia cromática) ---
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    color_hist = []
    for ch in range(3):
        h = cv2.calcHist([hsv], [ch], None, [_COLOR_BINS], [0, 256]).flatten()
        h = h / (h.sum() + 1e-7)  # normalizar a distribución
        color_hist.append(h)
    color_vec = np.concatenate(color_hist)

    return np.concatenate([hog_vec, color_vec]).astype(np.float32)


def build_feature_matrix(df, cache_name: str, use_cache: bool = True):
    """Construye (y cachea) la matriz X y el vector y para un manifest.

    `df` debe tener columnas `abs_path` y `quality_idx`.
    Devuelve (X, y) como arreglos numpy.
    """
    cache_x = PROCESSED_DIR / f"X_{cache_name}.npy"
    cache_y = PROCESSED_DIR / f"y_{cache_name}.npy"
    if use_cache and cache_x.exists() and cache_y.exists():
        return np.load(cache_x), np.load(cache_y)

    feats, labels = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"features[{cache_name}]"):
        img = load_image_rgb(row["abs_path"])
        if img is None:
            continue
        feats.append(extract_features(img))
        labels.append(int(row["quality_idx"]))

    X = np.vstack(feats)
    y = np.asarray(labels)
    np.save(cache_x, X)
    np.save(cache_y, y)
    return X, y
