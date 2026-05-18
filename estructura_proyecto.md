# Estructura del Proyecto — FruitVision

## Metodología: CRISP-DM Aplicada

Este documento describe cómo se aplica la metodología CRISP-DM al proyecto de clasificación de calidad de frutas y verduras.

---

## Fase 1: Comprensión del Negocio

**Objetivo del negocio:**  
Reducir las pérdidas económicas y el desperdicio de alimentos causados por la clasificación manual subjetiva en mercados y agroindustrias de Colombia.

**Objetivo de minería de datos:**  
Construir un clasificador de imágenes que asigne automáticamente una categoría de calidad (Premium / Estándar / Descarte) y una estimación de tamaño (Pequeño / Mediano / Grande) a frutas y verduras individuales.

**Criterio de éxito:**  
- Accuracy ≥ 85% en el conjunto de prueba para clasificación de calidad.
- F1-Score macro ≥ 0.80 considerando el desbalanceo entre clases.

**Preguntas clave:**
- ¿Cuáles son los atributos visuales más relevantes para determinar la calidad?
- ¿Qué tipos de defectos son los más frecuentes y qué impacto tienen en el precio?
- ¿Cómo varía la percepción de calidad según el tipo de fruta/verdura?

---

## Fase 2: Comprensión de los Datos

**Fuentes de datos:**

| Fuente | Tipo | Cantidad estimada | Formato |
|--------|------|-------------------|---------|
| Kaggle - Fruit Quality Classification | Pública | ~1500 imágenes | JPG/PNG |
| Recolección propia (plazas, supermercados) | Propia | 30–50 imágenes | JPG |
| Carpeta `mixed_quality` del dataset | Pública | Variable | JPG/PNG |

**Análisis exploratorio (EDA):**
- Distribución de clases por categoría de calidad.
- Distribución de tamaños.
- Análisis de balance de clases.
- Verificación de calidad de imágenes (resolución, iluminación, fondo).
- Variabilidad dentro de cada clase.

📓 Ver: `notebooks/2_data_understanding/EDA.ipynb`

---

## Fase 3: Preparación de los Datos

**Pasos de preprocesamiento:**

1. **Limpieza:** Eliminación de imágenes duplicadas o corruptas.
2. **Segmentación:** Recorte de frutas individuales cuando hay varias por foto.
3. **Redimensionamiento:** Todas las imágenes a 224×224 píxeles.
4. **Normalización:** Valores de píxel al rango [0, 1] o media/desviación estándar de ImageNet.
5. **Data Augmentation:**
   - Rotación aleatoria (±15°)
   - Flip horizontal
   - Ajuste de brillo y contraste
   - Zoom aleatorio (±10%)
6. **División del dataset:**
   - 70% entrenamiento
   - 15% validación
   - 15% prueba

**Manejo de desbalanceo:**
- Técnica a definir: oversampling (SMOTE en espacio de características), pesos de clase, o aumento de datos específico por clase.

📓 Ver: `notebooks/3_data_preparation/preprocessing.ipynb`

---

## Fase 4: Modelado

### Modelos seleccionados

**Machine Learning tradicional (con extracción de características):**

| Modelo | Características | Hiperparámetros a ajustar |
|--------|----------------|--------------------------|
| SVM | HOG + Color Histograms | C, kernel, gamma |
| Random Forest | HOG + LBP | n_estimators, max_depth, min_samples_split |

**Deep Learning:**

| Modelo | Arquitectura | Parámetros |
|--------|-------------|-----------|
| CNN desde cero | 3 capas conv + pooling + dense | lr, batch_size, dropout |

**Ajuste de hiperparámetros:**
- ML tradicional: GridSearchCV con validación cruzada (k=5).
- CNN: Búsqueda manual + callbacks (EarlyStopping, ReduceLROnPlateau).

📓 Ver: `notebooks/4_modeling/`

---

## Fase 5: Evaluación

**Métricas principales:**

| Métrica | Justificación |
|---------|--------------|
| Accuracy | Rendimiento general del clasificador |
| F1-Score (macro) | Manejo de clases desbalanceadas |
| Precision / Recall por clase | Identificar clases problemáticas |
| Matriz de confusión | Análisis de errores entre clases |
| Curva ROC / AUC | Análisis de umbral de decisión |

**Línea base (baseline):**
- Clasificador por mayoría: predice siempre la clase más frecuente.
- Se compararán todos los modelos contra esta línea base.

**Análisis de errores:**
- ¿Qué tipos de fruta confunde el modelo con mayor frecuencia?
- ¿Los errores son sistemáticos (siempre confunde Premium con Estándar)?

📓 Ver: `notebooks/5_evaluation/model_evaluation.ipynb`

---

## Fase 6: Despliegue

**Interfaz gráfica:**
- Framework: **Streamlit** (aplicación web ligera)
- Funcionalidades:
  - Carga de imagen desde disco
  - Captura en tiempo real con cámara web
  - Visualización de la predicción (clase + tamaño + confianza)
  - Bounding box sobre la fruta detectada

**Flujo de la aplicación:**

```
[Imagen de entrada]
        ↓
[Preprocesamiento]
        ↓
[Modelo entrenado]
        ↓
[Clase de calidad + Tamaño]
        ↓
[Visualización en interfaz]
```

📓 Ver: `src/app/app.py`

---

## Diagrama CRISP-DM personalizado

```
┌──────────────────────────────────────────────────────────────────┐
│                        CRISP-DM — FruitVision                    │
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     │
│  │  1. Negocio  │────▶│  2. Datos    │────▶│  3. Prep.    │     │
│  │              │     │   (EDA)      │     │   Datos      │     │
│  │ Clasificar   │     │ ~1500 imgs   │     │ Augment.     │     │
│  │ calidad de   │     │ + 50 propias │     │ Resize 224px │     │
│  │ frutas/verd. │     │ EDA, balance │     │ Train/Val/   │     │
│  └──────────────┘     └──────────────┘     │ Test split   │     │
│                                            └──────┬───────┘     │
│                                                   │             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────▼───────┐     │
│  │  6. Despl.   │◀────│  5. Eval.    │◀────│  4. Modelado │     │
│  │              │     │              │     │              │     │
│  │ Streamlit    │     │ F1, Accuracy │     │ SVM          │     │
│  │ Cámara web   │     │ Conf. Matrix │     │ RandomForest │     │
│  │ Pred. en     │     │ ROC-AUC      │     │ CNN (3 conv) │     │
│  │ tiempo real  │     │ vs baseline  │     │ GridSearch   │     │
│  └──────────────┘     └──────────────┘     └──────────────┘     │
└──────────────────────────────────────────────────────────────────┘
```

---

## Plan de trabajo y cronograma

| Semana | Actividad | Responsable |
|--------|-----------|-------------|
| 1–2 | Comprensión del negocio y datos (EDA) | Todos |
| 3–4 | Recolección de imágenes propias + anotación | Todos |
| 5–6 | Preprocesamiento y preparación del dataset | Integrante 1 |
| 7–8 | Entrenamiento modelos ML tradicionales | Integrante 2 |
| 9–10 | Entrenamiento CNN | Integrante 3 |
| 11 | Evaluación y comparación de modelos | Todos |
| 12 | Desarrollo de interfaz gráfica | Todos |
| 13 | Informe final y video | Todos |
