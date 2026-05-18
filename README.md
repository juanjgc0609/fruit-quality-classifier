# 🍎 FruitVision: Sistema de Clasificación de Calidad de Frutas y Verduras

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CRISP-DM](https://img.shields.io/badge/Methodology-CRISP--DM-orange.svg)]()
[![Universidad Icesi](https://img.shields.io/badge/Universidad-Icesi-red.svg)]()

> Proyecto Final — Algoritmos y Programación III · Semestre 2026-1  
> Facultad de Ingeniería, Diseño y Ciencias Aplicadas · Departamento de Computación y Sistemas Inteligentes

---

## 📋 Descripción

Sistema automático de clasificación de calidad de frutas y verduras basado en visión por computadora. A partir de una imagen estática o captura en tiempo real, el sistema predice:

- **Clase de calidad** (Premium / Estándar / Descarte)
- **Estimación de tamaño** (Pequeño / Mediano / Grande)

El sistema fue desarrollado aplicando la metodología **CRISP-DM** y combinando modelos de machine learning tradicional con redes neuronales convolucionales (CNN).

---

## 👥 Integrantes del grupo

| Nombre | Código | GitHub |
|--------|--------|--------|
| Juan José Gordillo Córdoba | — | [@juanjgc0609](https://github.com/juanjgc0609) |
| Sebastián Jiménez Galvis | — | — |
| Juan Pablo Zambrano Cortez | — | — |

---

## 🗂️ Estructura del repositorio

```
fruit-quality-classifier/
│
├── data/
│   ├── raw/                    # Imágenes originales sin procesar
│   ├── processed/              # Imágenes preprocesadas y aumentadas
│   ├── external/               # Dataset Kaggle (Fruit Quality Classification)
│   └── annotations/            # Etiquetas CSV de calidad y tamaño
│
├── notebooks/
│   ├── 1_business_understanding/   # Fase 1 CRISP-DM
│   ├── 2_data_understanding/       # Fase 2 CRISP-DM - EDA
│   ├── 3_data_preparation/         # Fase 3 CRISP-DM - Preprocesamiento
│   ├── 4_modeling/                 # Fase 4 CRISP-DM - Entrenamiento de modelos
│   ├── 5_evaluation/               # Fase 5 CRISP-DM - Evaluación y comparación
│   └── 6_deployment/               # Fase 6 CRISP-DM - Despliegue
│
├── src/
│   ├── data/                   # Scripts de carga y procesamiento de datos
│   ├── features/               # Extracción de características
│   ├── models/                 # Definición y entrenamiento de modelos
│   ├── visualization/          # Gráficos y visualizaciones
│   └── app/                    # Interfaz gráfica (Streamlit/Tkinter)
│
├── models/
│   ├── saved/                  # Modelos entrenados (.pkl, .h5, .pt)
│   └── checkpoints/            # Checkpoints durante entrenamiento
│
├── reports/
│   ├── figures/                # Gráficas y visualizaciones exportadas
│   └── final/                  # Informe final en PDF
│
├── references/                 # Artículos y materiales de referencia
│
├── requirements.txt
├── environment.yml
├── estructura_proyecto.md
└── README.md
```

---

## 🔬 Metodología: CRISP-DM

```
┌─────────────────────────────────────────────────┐
│                  CRISP-DM                        │
│                                                  │
│  1. Comprensión   →  2. Comprensión   →  3. Prep.│
│     del Negocio       de los Datos       de Datos│
│                                                  │
│  6. Despliegue   ←  5. Evaluación  ←  4. Modelado│
└─────────────────────────────────────────────────┘
```

| Fase | Descripción | Notebook |
|------|-------------|----------|
| 1. Comprensión del negocio | Impacto económico, objetivos | `notebooks/1_business_understanding/` |
| 2. Comprensión de los datos | EDA, distribución de clases, calidad | `notebooks/2_data_understanding/` |
| 3. Preparación de los datos | Augmentation, normalización, split | `notebooks/3_data_preparation/` |
| 4. Modelado | SVM, Random Forest, CNN, transfer learning | `notebooks/4_modeling/` |
| 5. Evaluación | Métricas, matriz de confusión, comparación | `notebooks/5_evaluation/` |
| 6. Despliegue | Interfaz gráfica en tiempo real | `notebooks/6_deployment/` |

---

## 🤖 Modelos implementados

El proyecto requiere mínimo **2 modelos de ML tradicional** y **1 de Deep Learning**.

**Machine Learning tradicional** (características extraídas: HOG + histogramas de color):

| Modelo | Hiperparámetros a ajustar | Accuracy (val) | F1-Score |
|--------|--------------------------|----------------|----------|
| Random Forest | n_estimators, max_depth, min_samples_split | — | — |
| XGBoost | learning_rate, max_depth, n_estimators, subsample | — | — |

**Deep Learning:**

| Modelo | Arquitectura | Accuracy (val) | F1-Score |
|--------|-------------|----------------|----------|
| CNN desde cero | 3 conv + MaxPooling + Dense + Dropout | — | — |

> Ajuste de hiperparámetros: GridSearchCV k=5 para ML · EarlyStopping + ReduceLROnPlateau para CNN.  
> Los resultados se actualizarán conforme avance el proyecto.

---

## 📦 Datos

- **Dataset principal:** [Fruit Quality Classification — Kaggle](https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification)
- **Imágenes propias:** 30–50 imágenes recolectadas en plazas y supermercados de Cali
- **Etiquetas:** Calidad (Premium / Estándar / Descarte) + Tamaño (Pequeño / Mediano / Grande)

---

## 🚀 Instalación y uso

### 1. Clonar el repositorio
```bash
git clone https://github.com/juanjgc0609/fruit-quality-classifier.git
cd fruit-quality-classifier
```

### 2. Crear el entorno virtual
```bash
conda env create -f environment.yml
conda activate fruit-quality
# O con pip:
pip install -r requirements.txt
```

### 3. Descargar los datos
```bash
# Instrucciones en data/README.md
```

### 4. Ejecutar la interfaz de demostración (Fase 6 — Despliegue)

La interfaz gráfica se desarrolla en la **Fase 6** del proyecto como parte del despliegue CRISP-DM. Permite cargar una imagen o usar la cámara y ver la predicción del modelo en tiempo real.

```bash
# Solo disponible una vez entrenados y guardados los modelos
streamlit run src/app/app.py
```

> **Streamlit** es una librería de Python que convierte un script en una app web sin necesidad de HTML/CSS. Es el framework elegido para el despliegue por su simplicidad.

---

## 📊 Resultados principales

> *Esta sección se completará con gráficas y métricas al finalizar el proyecto.*

---

## ⚖️ Consideraciones éticas

- Posible sesgo en las etiquetas de calidad si reflejan estándares no inclusivos.
- El sistema no debe usarse como única herramienta de decisión sin supervisión humana.
- Los datos recolectados no incluyen información personal de vendedores o compradores.

---

## 📄 Informe final

El informe final está disponible en [`reports/final/`](reports/final/).

---

## 🎥 Video de presentación

> *Enlace al video — máximo 10 minutos.*

---

## 📚 Referencias

- Ryan Park. *Fruit Quality Classification Dataset*. Kaggle, 2023.
- Wirth, R., & Hipp, J. (2000). *CRISP-DM: Towards a standard process model for data mining*.
- Lecun, Y., et al. (1998). *Gradient-based learning applied to document recognition*. Proceedings of the IEEE.

---

*Proyecto desarrollado en el marco del curso Algoritmos y Programación III — Universidad Icesi, Cali, Colombia.*
