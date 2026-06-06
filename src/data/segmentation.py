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


# ─────────────────────────────────────────────────────────────────────────────
# Segmentación MULTI-fruta + estimación de daño (fusión con el trabajo del
# compañero). Se usa para la carpeta Mixed, que el enunciado pide segmentar en
# frutas individuales. El daño re-etiqueta cada recorte por la heurística NTC-4580.
# ─────────────────────────────────────────────────────────────────────────────

def _foreground_mask(gray: np.ndarray, adaptive: bool) -> np.ndarray:
    """Máscara binaria de primer plano (Otsu o adaptativo) + morfología."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    if adaptive:
        mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 31, 2)
    else:
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def compute_damage_pct(crop_rgb: np.ndarray, mask: np.ndarray) -> float:
    """% de píxeles de la fruta con signos de daño (heurística NTC-4580, en HSV).

    Combina tres evidencias dentro de la máscara de la fruta:
      - oscuros (V<65)            -> necrosis / podredumbre
      - bajo cromatismo (S<55,V<180) -> falta de pigmento / zonas enfermas
      - pardos (H<35,30<S<190,V<170) -> oxidación / magulladuras
    """
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    fruit = mask > 0
    n = int(fruit.sum())
    if n == 0:
        return 100.0
    dark = (v < 65) & fruit
    low_sat = (s < 55) & (v < 180) & fruit
    brownish = (h < 35) & (s > 30) & (s < 190) & (v < 170) & fruit
    return float(100.0 * (dark | low_sat | brownish).sum() / n)


def assign_quality_by_damage(damage_pct: float, premium_max: float = 2.0,
                             standard_max: float = 18.0) -> str:
    """Mapea el % de daño a clase de calidad (solo para recortes de Mixed)."""
    if damage_pct < premium_max:
        return "Premium"
    if damage_pct <= standard_max:
        return "Estándar"
    return "Descarte"


def segment_instances(image_rgb: np.ndarray, allow_multiple: bool = True,
                      min_area_ratio: float = 0.01):
    """Segmenta una o varias frutas en una imagen (fondo posiblemente complejo).

    Elige entre Otsu y umbral adaptativo el que produce contornos de mayor área
    media (penaliza fragmentación). Devuelve una lista de dicts con:
      crop_rgb, mask (del recorte), diameter_norm (por diagonal de imagen original),
      damage_pct.
    """
    h, w = image_rgb.shape[:2]
    img_diag = float(np.hypot(h, w))
    min_area_px = h * w * min_area_ratio
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    best = None
    for adaptive in (False, True):
        mask = _foreground_mask(gray, adaptive)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in cnts if cv2.contourArea(c) >= min_area_px]
        score = float(np.mean([cv2.contourArea(c) for c in valid])) if valid else 0.0
        if best is None or score > best[0]:
            best = (score, valid)
    contours = sorted(best[1], key=cv2.contourArea, reverse=True)
    if not allow_multiple:
        contours = contours[:1]

    out = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area <= 0:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        pad = max(2, int(0.02 * max(h, w)))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        crop = image_rgb[y0:y1, x0:x1].copy()
        if crop.size == 0:
            continue
        full = np.zeros((h, w), np.uint8)
        cv2.drawContours(full, [cnt], -1, 255, thickness=cv2.FILLED)
        mcrop = full[y0:y1, x0:x1]
        out.append({
            "crop_rgb": crop,
            "mask": mcrop,
            "diameter_norm": float(2.0 * np.sqrt(area / np.pi)) / img_diag,
            "damage_pct": compute_damage_pct(crop, mcrop),
        })
    return out
