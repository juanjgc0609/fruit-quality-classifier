"""
Detección de casi-duplicados (anti-fuga de datos)
=================================================
El dataset Kaggle contiene ráfagas de fotos de la MISMA fruta física (tomadas
con segundos de diferencia). Si esas fotos casi idénticas caen unas en train y
otras en test, los modelos "memorizan" frutas concretas en vez de generalizar,
inflando las métricas (fuga de datos / data leakage).

Este módulo asigna a cada imagen un `group_id` tal que todas las fotos
casi-idénticas comparten grupo. Luego el split se hace POR GRUPO (todas las
fotos de una fruta van a la misma partición), eliminando la fuga.

Método: dHash perceptual de 64 bits + clustering por distancia de Hamming.
"""

from __future__ import annotations

import cv2
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from tqdm.auto import tqdm

# Umbral de Hamming para considerar dos imágenes "la misma fruta".
# 5 captura ráfagas/recompresiones sin fusionar frutas genuinamente distintas.
HAMMING_THR: int = 5


def dhash(path, size: int = 8) -> int | None:
    """Huella perceptual de 64 bits (gradiente horizontal en gris 9x8)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (size + 1, size))
    diff = img[:, 1:] > img[:, :-1]
    bits = 0
    for b in diff.flatten():
        bits = (bits << 1) | int(b)
    return bits


def _popcount(x: np.ndarray) -> np.ndarray:
    """popcount vectorizado para uint64."""
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    return (x * 0x0101010101010101) >> 56


def assign_groups(paths, thr: int = HAMMING_THR) -> np.ndarray:
    """Devuelve un array de `group_id` (int) por imagen.

    Imágenes con distancia de Hamming <= thr quedan en el mismo grupo
    (componentes conexas). Las imágenes únicas forman su propio grupo.
    """
    hashes = [dhash(p) for p in tqdm(paths, desc="dhash")]
    valid = np.array([h is not None for h in hashes])
    H = np.array([h if h is not None else 0 for h in hashes], dtype=np.uint64)
    n = len(H)

    rows, cols = [], []
    for i in range(n):
        d = _popcount(H ^ H[i]).astype(np.int64)
        d[i] = 999
        j = np.where(d <= thr)[0]
        rows += [i] * len(j)
        cols += list(j)
    g = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, labels = connected_components(g, directed=False)

    # Las imágenes ilegibles quedan como grupos propios (no se fusionan).
    labels = labels.copy()
    next_id = labels.max() + 1
    for i in range(n):
        if not valid[i]:
            labels[i] = next_id
            next_id += 1
    return labels
