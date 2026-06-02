"""
Resolución de rutas e I/O de imágenes
=====================================
El `labels.csv` fue generado en Windows y guarda rutas relativas con
backslashes y prefijos `..\\..\\` (relativas a la ubicación del notebook de
EDA). Este módulo normaliza esas rutas a rutas absolutas válidas en cualquier
sistema operativo, tomando como ancla la raíz del repositorio.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from src.config import PROJECT_ROOT

# Prefijos de subida de directorio: "../", "..\" repetidos al inicio.
_LEADING_UPDIRS = re.compile(r"^(\.\.[\\/])+")


def resolve_path(raw_path: str) -> Path:
    """Convierte una ruta del labels.csv en una ruta absoluta del repo.

    Ejemplo:
        '..\\..\\data\\external\\Good Quality_Fruits\\Apple_Good\\img.jpg'
        -> {PROJECT_ROOT}/data/external/Good Quality_Fruits/Apple_Good/img.jpg
    """
    p = str(raw_path).replace("\\", "/")      # backslashes -> slashes
    p = _LEADING_UPDIRS.sub("", p)            # quitar '../' iniciales
    return (PROJECT_ROOT / p).resolve()


def load_image_rgb(path) -> np.ndarray | None:
    """Carga una imagen como arreglo RGB (uint8). Devuelve None si falla.

    Acepta tanto una ruta cruda del CSV como una ruta ya resuelta.
    """
    path = Path(path)
    if not path.is_absolute():
        path = resolve_path(str(path))
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
