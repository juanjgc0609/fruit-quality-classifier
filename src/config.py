"""
Configuración central del proyecto FruitVision
==============================================
Punto único de verdad para rutas, constantes y parámetros reproducibles.
Importar desde cualquier notebook/módulo:

    from src.config import (PROJECT_ROOT, QUALITY_CLASSES, SEED, ...)

Todas las rutas son absolutas y se derivan de la ubicación de este archivo,
de modo que funcionan igual desde un notebook (sin importar su profundidad)
o desde un script.
"""

from __future__ import annotations

from pathlib import Path

# ─── Rutas base ───────────────────────────────────────────────────────────────
# src/config.py -> parents[1] es la raíz del repositorio.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
EXTERNAL_DIR = DATA_DIR / "external"          # Dataset Kaggle (no versionado)
PROCESSED_DIR = DATA_DIR / "processed"        # Manifests y features cacheados
ANNOTATIONS_DIR = DATA_DIR / "annotations"    # labels.csv, labels_own.csv

MODELS_DIR = PROJECT_ROOT / "models" / "saved"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

LABELS_CSV = ANNOTATIONS_DIR / "labels.csv"

# Crear directorios de salida si no existen (idempotente).
for _d in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─── Reproducibilidad ───────────────────────────────────────────────────────────
SEED: int = 42

# ─── Clases del problema ────────────────────────────────────────────────────────
# Calidad: 3 categorías (cumple el mínimo "3 o más" del enunciado).
#   Premium  <- Good Quality_Fruits
#   Estándar <- Mixed Qualit_Fruits
#   Descarte <- Bad Quality_Fruits
QUALITY_CLASSES: list[str] = ["Premium", "Estándar", "Descarte"]
QUALITY_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(QUALITY_CLASSES)}

# Tamaño: derivado por segmentación (diámetro en píxeles normalizados).
SIZE_CLASSES: list[str] = ["Pequeño", "Mediano", "Grande"]

# ─── Parámetros de preparación de datos ─────────────────────────────────────────
# Cap por clase de calidad para mitigar el desbalanceo 10:1 (Premium domina).
# La clase minoritaria (Estándar ~1074) se conserva completa; el residual se
# corrige con class_weight='balanced' en los modelos.
CAP_PER_QUALITY: int = 1500

# Split estratificado por calidad.
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

# ─── Tamaños de imagen ──────────────────────────────────────────────────────────
FEATURE_IMG_SIZE: tuple[int, int] = (128, 128)  # Para HOG + histogramas (ML)
CNN_IMG_SIZE: tuple[int, int] = (96, 96)         # Entrada de la CNN (CPU-friendly)
