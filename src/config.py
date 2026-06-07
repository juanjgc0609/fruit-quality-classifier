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

# Rutas base
# src/config.py -> parents[1] es la raíz del repositorio.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
EXTERNAL_DIR = DATA_DIR / "external"          # Dataset Kaggle (no versionado)
RAW_DIR = DATA_DIR / "raw"                    # Dataset propio (recolección del grupo)
PROCESSED_DIR = DATA_DIR / "processed"        # Manifests y features cacheados
ANNOTATIONS_DIR = DATA_DIR / "annotations"    # labels.csv, labels_own.csv

MODELS_DIR = PROJECT_ROOT / "models" / "saved"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

LABELS_CSV = ANNOTATIONS_DIR / "labels.csv"

# Crear directorios de salida si no existen (idempotente).
for _d in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Reproducibilidad
SEED: int = 42

# Clases del problema
# Calidad: 3 categorías (cumple el mínimo "3 o más" del enunciado).
# Premium  <- Good Quality_Fruits
# Estándar <- Mixed Qualit_Fruits
# Descarte <- Bad Quality_Fruits
QUALITY_CLASSES: list[str] = ["Premium", "Estándar", "Descarte"]
QUALITY_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(QUALITY_CLASSES)}

# Mapeo de carpetas de calidad -> clase del proyecto.
# Good    -> Premium       (1 fruta por foto)
# Regular -> Estándar      (1 fruta por foto, calidad media) [solo dataset propio]
# Bad     -> Descarte      (1 fruta por foto)
# La carpeta Kaggle "Mixed" se EXCLUYE del entrenamiento (varias frutas por foto,
# fondo no uniforme); se reserva para el ejercicio de segmentación/evaluación.
FOLDER_QUALITY_MAP: dict[str, str] = {
    "Good": "Premium",
    "Regular": "Estándar",
    "Bad": "Descarte",
}

# Tamaño: derivado por segmentación (diámetro en píxeles normalizados).
SIZE_CLASSES: list[str] = ["Pequeño", "Mediano", "Grande"]

# Parámetros de preparación de datos
# Cap por (fruta × calidad): evita que una sola fruta domine una clase
# (p. ej. Pomegranate_Good inflaba Premium, hallazgo del EDA §2.3) y reduce el
# desbalanceo. El residual se corrige con class_weight='balanced'.
CAP_PER_FRUIT_QUALITY: int = 400

# Enriquecimiento con la carpeta Mixed (segmentación multi-fruta)
# La carpeta Kaggle "Mixed" tiene varias frutas por foto. La segmentamos en
# recortes individuales y los re-etiquetamos por daño (NTC-4580). Se usan SOLO
# como enriquecimiento de TRAIN (nunca val/test) para no introducir etiquetas
# derivadas de color en la evaluación (evita métricas circulares).
INCLUDE_MIXED_ENRICHMENT: bool = True
MIXED_DIRNAME: str = "Mixed Qualit_Fruits"          # nombre real en el dataset Kaggle
CAP_MIXED_PER_FRUIT_QUALITY: int = 150              # tope de recortes Mixed por fruta×clase
DAMAGE_PREMIUM_MAX: float = 2.0                      # umbral de daño (%) para Premium
DAMAGE_STANDARD_MAX: float = 18.0                   # umbral de daño (%) para Estándar
MIXED_MIN_AREA_RATIO: float = 0.01                  # área mínima de contorno (fracción)

# Split estratificado por calidad.
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

# Tamaños de imagen
FEATURE_IMG_SIZE: tuple[int, int] = (128, 128)  # Para HOG + histogramas (ML)
CNN_IMG_SIZE: tuple[int, int] = (96, 96)         # Entrada de la CNN (CPU-friendly)
