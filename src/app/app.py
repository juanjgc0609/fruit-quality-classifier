"""
FruitVision — Fase 6: Despliegue (interfaz gráfica)
===================================================
App web (Streamlit) que clasifica la **calidad** de una fruta y estima su
**tamaño** a partir de una imagen subida o capturada con la cámara.

Conecta los modelos entrenados en la Fase 4:
  - Calidad: XGBoost (HOG+color) ó CNN desde cero  (elegible en la barra lateral)
  - Tamaño:  segmentación (diámetro normalizado) + umbrales aprendidos en train

Uso:
    conda activate fruit-quality
    streamlit run src/app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

#  Bootstrap: raíz del repo en el path para importar `src`
ROOT = Path(__file__).resolve()
while not (ROOT / "src").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image

from src.config import CNN_IMG_SIZE, MODELS_DIR, QUALITY_CLASSES
from src.data.preprocessing import load_size_thresholds
from src.data.segmentation import assign_size_class, segment_fruit
from src.features.extract import extract_features

#  Configuración de la página
st.set_page_config(page_title="FruitVision", page_icon="🍎", layout="wide")

QUALITY_ICON = {"Premium": "🟢", "Estándar": "🟡", "Descarte": "🔴"}
RECOMMENDATION = {
    "Premium": ("success", "✅ Apto para venta directa (primera calidad)."),
    "Estándar": ("warning", "⚠️ Apto con descuento o para procesamiento."),
    "Descarte": ("error", "❌ No apto para consumo directo."),
}


#  Carga de modelos (cacheada)
@st.cache_resource
def load_quality_models():
    """Carga los modelos de calidad y los umbrales de tamaño una sola vez."""
    ml = joblib.load(MODELS_DIR / "best_quality_ml.pkl")          # XGBoost
    cnn = None
    keras_path = MODELS_DIR / "cnn_quality.keras"
    if keras_path.exists():
        import tensorflow as tf
        cnn = tf.keras.models.load_model(keras_path)
    thresholds = load_size_thresholds()
    return ml, cnn, thresholds


#  Inferencia
def predict_quality(image_rgb: np.ndarray, model_name: str, ml, cnn):
    """Devuelve (clase_calidad, confianza, vector_probabilidades)."""
    if model_name == "CNN (desde cero)" and cnn is not None:
        x = (cv2.resize(image_rgb, CNN_IMG_SIZE).astype("float32") / 255.0)[None]
        proba = cnn.predict(x, verbose=0)[0]
    else:  # XGBoost (HOG + color)
        feats = extract_features(image_rgb).reshape(1, -1)
        proba = ml.predict_proba(feats)[0]
    idx = int(np.argmax(proba))
    return QUALITY_CLASSES[idx], float(proba[idx]), proba


def analyze(image_rgb: np.ndarray, model_name: str, ml, cnn, thresholds):
    """Pipeline completo: segmenta, estima tamaño, predice calidad y dibuja bbox."""
    seg = segment_fruit(image_rgb)
    size_class = assign_size_class(seg.diameter_norm, thresholds)
    quality, conf, proba = predict_quality(image_rgb, model_name, ml, cnn)

    vis = image_rgb.copy()
    x, y, w, h = seg.bbox
    thick = max(2, image_rgb.shape[1] // 200)
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), thick)
    return dict(quality=quality, conf=conf, proba=proba, size=size_class,
                diameter=seg.diameter_norm, vis=vis)


#  Interfaz
def main():
    st.title("🍎 FruitVision — Clasificación de Calidad de Frutas")
    st.caption("Fase 6 (Despliegue) · Algoritmos y Programación III · Universidad Icesi")

    try:
        ml, cnn, thresholds = load_quality_models()
    except FileNotFoundError:
        st.error("No se encontraron los modelos en `models/saved/`. "
                 "Ejecuta primero las Fases 3 y 4 (notebooks).")
        st.stop()

    # Barra lateral
    with st.sidebar:
        st.header("⚙️ Configuración")
        model_options = ["XGBoost (HOG + color)"]
        if cnn is not None:
            model_options.append("CNN (desde cero)")
        model_name = st.radio("Modelo de calidad", model_options)
        mode = st.radio("Entrada", ["📁 Subir imagen", "📷 Cámara"])
        st.markdown("---")
        st.markdown("**Clases de calidad**")
        for q in QUALITY_CLASSES:
            st.markdown(f"{QUALITY_ICON[q]} {q}")
        st.caption("Tamaño = diámetro de la fruta relativo al encuadre "
                   "(fracción de la imagen), no diámetro físico.")

    # Entrada de imagen
    image = None
    if mode == "📁 Subir imagen":
        up = st.file_uploader("Carga una foto de una fruta (fondo simple)",
                              type=["jpg", "jpeg", "png"])
        if up:
            image = Image.open(up)
    else:
        cam = st.camera_input("Toma una foto con la cámara")
        if cam:
            image = Image.open(cam)

    if image is None:
        st.info("Sube una imagen o toma una foto para ver la predicción.")
        return

    image_rgb = np.array(image.convert("RGB"))
    with st.spinner("Analizando…"):
        r = analyze(image_rgb, model_name, ml, cnn, thresholds)

    col1, col2 = st.columns(2)
    with col1:
        st.image(r["vis"], caption="Fruta detectada (bounding box)",
                 use_container_width=True)
    with col2:
        st.subheader("Resultados")
        st.markdown(f"### {QUALITY_ICON[r['quality']]} Calidad: **{r['quality']}**")
        st.progress(r["conf"], text=f"Confianza: {r['conf']:.1%}")
        st.markdown(f"### 📏 Tamaño: **{r['size']}**")
        st.caption(f"Diámetro normalizado = {r['diameter']:.3f} (área/diagonal)")

        kind, msg = RECOMMENDATION[r["quality"]]
        getattr(st, kind)(msg)

        with st.expander("Detalle de probabilidades por clase"):
            for q, p in zip(QUALITY_CLASSES, r["proba"]):
                st.write(f"{QUALITY_ICON[q]} {q}: {p:.1%}")

    st.markdown("---")
    st.caption(f"Modelo de calidad: {model_name} · "
               "Tamaño por segmentación (OpenCV).")


if __name__ == "__main__":
    main()
