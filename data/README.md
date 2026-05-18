# Datos del Proyecto

## Estructura

```
data/
├── raw/            # Imágenes originales — NO modificar
├── processed/      # Imágenes preprocesadas (224x224, normalizadas)
├── external/       # Dataset de Kaggle
└── annotations/    # Archivos CSV con etiquetas
    ├── labels.csv
    └── labels_own.csv
```

## Fuentes de datos

### 1. Dataset principal — Kaggle
**Fruit Quality Classification**  
URL: https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification

**Instrucciones de descarga:**
```bash
# Requiere kaggle CLI configurado
kaggle datasets download -d ryandpark/fruit-quality-classification
unzip fruit-quality-classification.zip -d data/external/
```

### 2. Imágenes propias recolectadas por el grupo

Cada imagen en `data/raw/` debe seguir el esquema de nombre:
```
{fruta}_{calidad}_{tamaño}_{id_grupo}_{numero}.jpg
# Ejemplo: manzana_premium_grande_G1_001.jpg
```

### 3. Formato del archivo de anotaciones `annotations/labels.csv`

| Campo | Descripción | Valores posibles |
|-------|-------------|-----------------|
| `filename` | Nombre del archivo | string |
| `fruit_type` | Tipo de fruta/verdura | manzana, tomate, papa, ... |
| `quality` | Clase de calidad | Premium, Estándar, Descarte |
| `size` | Estimación de tamaño | Pequeño, Mediano, Grande |
| `source` | Origen de la imagen | kaggle, propio |
| `annotator` | Quien realizó la anotación | string |

## Criterios de etiquetado

### Clase de calidad
| Clase | Descripción |
|-------|-------------|
| **Premium** | Sin defectos visibles, color uniforme, forma regular |
| **Estándar** | Defectos menores (manchas pequeñas), apta para consumo |
| **Descarte** | Deterioro visible, golpes severos, podredumbre |

### Clase de tamaño
| Clase | Descripción aproximada |
|-------|----------------------|
| **Pequeño** | Diámetro < 5 cm (en imagen normalizada) |
| **Mediano** | Diámetro 5–8 cm |
| **Grande** | Diámetro > 8 cm |

> **Nota:** Los datos originales de Kaggle NO se suben al repositorio por tamaño. Solo se sube `labels.csv`.
