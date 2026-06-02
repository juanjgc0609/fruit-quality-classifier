"""
C1 — Segmentación de fruta y estimación de tamaño
=================================================
El dataset Kaggle NO trae etiquetas de tamaño. El enunciado permite reportar el
tamaño como "diámetro en píxeles normalizados". Aquí lo resolvemos de forma NO
supervisada:

  1. Segmentamos la fruta del fondo (Otsu sobre la saturación HSV + morfología).
  2. Tomamos el mayor contorno (la fruta sobre fondo simple) y calculamos su
     diámetro equivalente en píxeles.
  3. Normalizamos por la diagonal de la imagen -> medida invariante a resolución.
  4. Discretizamos en {Pequeño, Mediano, Grande} usando terciles aprendidos de la
     distribución del propio dataset (ver `compute_size_thresholds`).

Este módulo cumple doble propósito:
  - Estimar el tamaño (salida requerida por el proyecto).
  - Recortar/segmentar frutas individuales (útil para la carpeta Mixed Quality,
    que el enunciado pide segmentar) y para el bounding box de la app.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SegmentationResult:
    """Resultado de segmentar una imagen."""
    mask: np.ndarray            # máscara binaria (uint8 0/255)
    bbox: tuple[int, int, int, int]  # (x, y, w, h) del mayor contorno
    diameter_px: float          # diámetro equivalente (círculo de igual área)
    diameter_norm: float        # diámetro normalizado por la diagonal de la imagen
    area_ratio: float           # fracción del área de imagen ocupada por la fruta


def segment_fruit(image_rgb: np.ndarray) -> SegmentationResult:
    """Segmenta la fruta dominante sobre un fondo simple.

    Estrategia robusta para fondos uniformes (claros u oscuros):
      - Umbral de Otsu sobre el canal de Saturación (la fruta suele ser más
        saturada que un fondo neutro) combinado con Otsu sobre escala de grises.
      - Operaciones morfológicas para cerrar huecos y eliminar ruido.
      - Selección del mayor componente conectado.
    """
    h, w = image_rgb.shape[:2]
    img_diag = float(np.hypot(h, w))

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Suavizado para reducir ruido de textura antes de umbralizar.
    sat = cv2.GaussianBlur(sat, (5, 5), 0)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu sobre saturación (objeto saturado vs fondo neutro).
    _, mask_sat = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu sobre gris, con inversión automática: nos quedamos con la región que
    # NO domina los bordes (heurística: el fondo toca los bordes de la imagen).
    _, mask_gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Unión de evidencias.
    mask = cv2.bitwise_or(mask_sat, mask_gray)

    # Si la "fruta" ocupa casi toda la imagen, probablemente segmentamos el
    # fondo; invertimos.
    if mask.mean() > 0.6 * 255:
        mask = cv2.bitwise_not(mask)

    # Morfología: cerrar huecos internos y abrir para quitar motas.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Mayor contorno.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback: imagen completa.
        return SegmentationResult(
            mask=np.full((h, w), 255, np.uint8),
            bbox=(0, 0, w, h),
            diameter_px=min(h, w),
            diameter_norm=min(h, w) / img_diag,
            area_ratio=1.0,
        )

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    x, y, bw, bh = cv2.boundingRect(largest)

    # Máscara limpia con solo el mayor contorno relleno.
    clean = np.zeros((h, w), np.uint8)
    cv2.drawContours(clean, [largest], -1, 255, thickness=cv2.FILLED)

    # Diámetro equivalente: diámetro de un círculo con la misma área.
    diameter_px = float(2.0 * np.sqrt(area / np.pi))

    return SegmentationResult(
        mask=clean,
        bbox=(x, y, bw, bh),
        diameter_px=diameter_px,
        diameter_norm=diameter_px / img_diag,
        area_ratio=float(area / (h * w)),
    )


def crop_to_object(image_rgb: np.ndarray, seg: SegmentationResult,
                   pad: float = 0.05) -> np.ndarray:
    """Recorta la imagen al bounding box de la fruta, con un pequeño padding."""
    h, w = image_rgb.shape[:2]
    x, y, bw, bh = seg.bbox
    px, py = int(bw * pad), int(bh * pad)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(w, x + bw + px), min(h, y + bh + py)
    return image_rgb[y0:y1, x0:x1]


def compute_size_thresholds(diameters_norm: np.ndarray) -> tuple[float, float]:
    """Aprende los cortes Pequeño/Mediano/Grande como terciles (33%, 66%).

    Devolver los cortes hace la asignación reproducible y data-driven, en vez de
    fijar umbrales arbitrarios. Se calcula una sola vez sobre el conjunto de
    entrenamiento y se reutiliza para val/test y en producción.
    """
    q33, q66 = np.quantile(diameters_norm, [1 / 3, 2 / 3])
    return float(q33), float(q66)


def assign_size_class(diameter_norm: float, thresholds: tuple[float, float]) -> str:
    """Asigna la clase de tamaño dado el diámetro normalizado y los cortes."""
    q33, q66 = thresholds
    if diameter_norm < q33:
        return "Pequeño"
    if diameter_norm < q66:
        return "Mediano"
    return "Grande"
