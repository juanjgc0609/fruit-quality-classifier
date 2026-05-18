"""
FruitVision — Aplicación de Clasificación de Calidad
=====================================================
Interfaz gráfica construida con Streamlit.

Uso:
    streamlit run src/app/app.py
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import joblib
import os

# ─── Configuración de la página ─────────────────────────────────────────────
st.set_page_config(
    page_title="FruitVision",
    page_icon="🍎",
    layout="centered",
)

# ─── Constantes ──────────────────────────────────────────────────────────────
QUALITY_LABELS = ["Premium", "Estándar", "Descarte"]
SIZE_LABELS = ["Pequeño", "Mediano", "Grande"]
QUALITY_COLORS = {"Premium": "🟢", "Estándar": "🟡", "Descarte": "🔴"}
MODEL_PATH = "models/saved/"

# ─── Carga de modelos ────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Carga los modelos entrenados desde disco."""
    # TODO: reemplazar con rutas reales al finalizar el entrenamiento
    # quality_model = joblib.load(os.path.join(MODEL_PATH, "best_quality_model.pkl"))
    # size_model = joblib.load(os.path.join(MODEL_PATH, "best_size_model.pkl"))
    return None, None  # Placeholder

quality_model, size_model = load_models()


# ─── Funciones de preprocesamiento ───────────────────────────────────────────
def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """Preprocesa una imagen PIL para ingresarla al modelo."""
    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalización
    return img


def predict(img_array: np.ndarray):
    """Realiza la predicción de calidad y tamaño."""
    # TODO: conectar con modelos reales
    # Por ahora retorna predicciones de ejemplo
    quality_idx = np.random.randint(0, 3)
    size_idx = np.random.randint(0, 3)
    quality_conf = np.random.uniform(0.7, 0.99)
    size_conf = np.random.uniform(0.7, 0.99)
    return (
        QUALITY_LABELS[quality_idx], quality_conf,
        SIZE_LABELS[size_idx], size_conf
    )


# ─── Interfaz principal ───────────────────────────────────────────────────────
def main():
    st.title("🍎 FruitVision")
    st.markdown("**Clasificación automática de calidad de frutas y verduras**")
    st.markdown("---")

    # Modo de entrada
    input_mode = st.radio(
        "Selecciona el modo de entrada:",
        ["📁 Subir imagen", "📷 Cámara en tiempo real"],
        horizontal=True,
    )

    image = None

    if input_mode == "📁 Subir imagen":
        uploaded_file = st.file_uploader(
            "Carga una imagen de fruta o verdura",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded_file:
            image = Image.open(uploaded_file)

    else:  # Cámara
        camera_image = st.camera_input("Toma una foto con tu cámara")
        if camera_image:
            image = Image.open(camera_image)

    # Predicción
    if image is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption="Imagen cargada", use_column_width=True)

        with col2:
            st.markdown("### Resultados")
            with st.spinner("Analizando imagen..."):
                img_array = preprocess_image(image)
                quality, q_conf, size, s_conf = predict(img_array)

            # Mostrar resultados
            st.markdown(f"**Clase de calidad:** {QUALITY_COLORS[quality]} {quality}")
            st.progress(q_conf, text=f"Confianza: {q_conf:.1%}")

            st.markdown(f"**Tamaño estimado:** 📏 {size}")
            st.progress(s_conf, text=f"Confianza: {s_conf:.1%}")

            # Recomendación
            st.markdown("---")
            if quality == "Premium":
                st.success("✅ Producto apto para venta directa.")
            elif quality == "Estándar":
                st.warning("⚠️ Producto apto con descuento o procesamiento.")
            else:
                st.error("❌ Producto no apto para consumo directo.")

    st.markdown("---")
    st.caption(
        "FruitVision · Algoritmos y Programación III · Universidad Icesi, 2026"
    )


if __name__ == "__main__":
    main()
