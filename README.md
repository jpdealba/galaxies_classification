# Introducción al Aprendizaje Automático: Clasificación Binaria de Estrellas y Galaxias

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/jupyter-notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Descripción del Proyecto

Este proyecto desarrolla un sistema de clasificación automática para distinguir entre **estrellas** y **galaxias** utilizando técnicas de aprendizaje automático aplicadas a datos fotométricos del **Sloan Digital Sky Survey (SDSS)**. 

La astronomía moderna enfrenta el desafío de procesar enormes volúmenes de datos provenientes de observaciones astronómicas. Con el advenimiento de telescopios de gran campo, la cantidad de objetos celestes detectados ha crecido exponencialmente, haciendo impracticable la clasificación manual. Este proyecto aborda este problema implementando y comparando múltiples algoritmos de machine learning.

## 🎯 Objetivos

### Objetivo Principal
Desarrollar un sistema de clasificación automática que permita distinguir eficientemente entre estrellas y galaxias utilizando técnicas de aprendizaje automático.

### Objetivos Específicos
- Implementar y comparar el rendimiento de **4 algoritmos** de clasificación binaria:
  - Random Forest
  - Regresión Logística  
  - Máquinas de Vectores de Soporte (SVM)
  - K-Nearest Neighbors (KNN)
- Realizar análisis exhaustivo de datos fotométricos
- Aplicar técnicas de preprocesamiento optimizadas
- Optimizar hiperparámetros mediante búsqueda aleatoria
- Alcanzar precisión superior al **90%** en clasificación

## 📊 Dataset

### Fuente de Datos
- **Origen**: Sloan Digital Sky Survey (SDSS)
- **Tamaño total**: ~5 millones de observaciones
  - **Entrenamiento**: 4 millones (2M estrellas + 2M galaxias)
  - **Prueba**: 1 millón de observaciones
- **Variables**: 50 características observacionales

### Estructura de Datos
```
├── Identificadores y Metadatos
│   ├── objID, run, camcol, field
├── Posición y Movimiento  
│   ├── ra, dec (coordenadas ecuatoriales)
│   ├── b, l (coordenadas galácticas)
│   └── rowv, colv (velocidades)
├── Magnitudes Fotométricas
│   ├── psfMag_* (magnitudes PSF en 5 filtros)
│   ├── u, g, r, i, z (magnitudes modelo)
│   └── modelFlux_* (flujos correspondientes)
├── Parámetros Morfológicos
│   ├── petroRad_* (radios petrosianos)
│   ├── expRad_* (radios exponenciales)
│   └── expAB_* (relaciones de ejes)
└── Parámetros de Stokes
    ├── q_* (polarización Q)
    └── u_* (polarización U)
```

## 🔧 Metodología

### 1. Exploración de Datos
- **Análisis de calidad**: Verificación de valores faltantes, duplicados
- **Estadísticas descriptivas**: Medidas de tendencia central y dispersión
- **Distribuciones**: Análisis de histogramas y detección de outliers
- **Correlaciones**: Matriz completa y correlación con variable objetivo

### 2. Preprocesamiento
```python
# Pipeline de preprocesamiento
├── Limpieza de datos
│   ├── Eliminación de identificadores no informativos
│   ├── Remoción de variables con alta proporción de ceros
│   └── Codificación de variable objetivo (star=0, galaxy=1)
├── Selección de características (13 variables finales)
│   ├── expRad_* (radios exponenciales)
│   ├── petroRad_* (radios petrosianos)  
│   └── g, r, i, z (magnitudes fotométricas)
└── Transformaciones
    ├── Transformación logarítmica (log1p)
    ├── Imputación con mediana
    └── Escalado robusto (RobustScaler)
```

### 3. Modelado
- **Algoritmos implementados**: Random Forest, Logistic Regression, SVM, KNN
- **Validación**: Validación cruzada estratificada (5-fold)
- **Optimización**: RandomizedSearchCV para hiperparámetros
- **Métricas**: Accuracy, F1-Score, Recall, Precision

## 📈 Resultados Principales

### Comparación de Modelos

| Modelo | Accuracy (CV) | F1-Score | Análisis de Overfitting |
|--------|---------------|----------|-------------------------|
| **Random Forest** | **95.71%** | 0.9571 | ⚠️ Overfitting severo (0% vs 4.29%) |
| **Logistic Regression** | 94.30% | 0.9430 | ✅ Excelente generalización (5.72% vs 5.70%) |
| **SVM** | 95.45% | 0.9545 | ⚠️ Ligero overfitting (4.25% vs 4.55%) |
| **KNN** | 94.25% | 0.9425 | ⚠️ Overfitting moderado (4.15% vs 5.75%) |

### Modelo Final Seleccionado: **Logistic Regression**

**Justificación de selección:**
- ✅ **Excelente capacidad de generalización** (sin overfitting)
- ✅ **Eficiencia computacional** superior
- ✅ **Interpretabilidad** de coeficientes
- ✅ **Escalabilidad** para grandes datasets

### Rendimiento Final
```
🎯 Accuracy en Test Set: 94.24%
📊 Matriz de Confusión:
    ├── Estrellas correctas: 942,122
    ├── Galaxias correctas: 942,614  
    ├── Falsos positivos: 57,390
    └── Falsos negativos: 57,874
```

## 🚀 Instalación y Uso

### Prerrequisitos
```bash
Python 3.8+
```

### Instalación de Dependencias
```bash
pip install -r requirements.txt
```

### Estructura de requirements.txt
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

### Uso Básico
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression

# Cargar datos
df = pd.read_csv('star_classification.csv')

# Características seleccionadas
features = [
    'expRad_r', 'expRad_i', 'expRad_g',
    'petroRad_r', 'expRad_z', 'petroRad_i',
    'petroRad_g', 'i', 'petroRad_z', 
    'expRad_u', 'z', 'r', 'g'
]

# Pipeline del modelo
pipeline = Pipeline([
    ("preprocessing", preprocessing_pipeline),
    ("classifier", LogisticRegression(
        penalty="l1",
        C=0.4655,
        l1_ratio=0.6075,
        solver="saga",
        max_iter=1000,
        random_state=42
    ))
])

# Entrenar modelo
X = df[features]
y = df['type_numeric']
pipeline.fit(X, y)

# Hacer predicciones
predictions = pipeline.predict(X_test)
```

## 📁 Estructura del Proyecto

```
proyecto/
├── documento/
│   ├── main.tex                    # Documento LaTeX principal
│   ├── out/
│   │   └── main.pdf               # Documento compilado
│   └── imagenes/                  # Figuras y gráficos
├── Proyecto (3).ipynb            # Notebook principal de implementación
├── matriz_correlacion_heatmap.xlsx # Análisis de correlaciones
├── README.md                      # Este archivo
└── requirements.txt               # Dependencias del proyecto
```

## 📊 Notebooks y Recursos

### Notebook Principal
🔗 **[Google Colab - Implementación Completa](https://colab.research.google.com/drive/1Z7cGOq95QmInkWO31x2LaOZ4O5ohhJIZ?usp=sharing)**

### Dataset
🔗 **[Kaggle - CelestialClassify Dataset](https://www.kaggle.com/datasets/hari31416/celestialclassify)**

## 🔍 Aspectos Técnicos Destacados

### Pipeline de Preprocesamiento Robusto
- **Transformación logarítmica**: Manejo de distribuciones sesgadas
- **RobustScaler**: Resistente a outliers astronómicos
- **Selección inteligente**: 13/50 variables más discriminativas

### Optimización de Hiperparámetros
```python
# RandomizedSearchCV configuración
param_distributions = {
    'classifier__penalty': ['l1', 'l2', 'elasticnet'],
    'classifier__C': uniform(0.001, 10),
    'classifier__l1_ratio': uniform(0, 1)
}
```

### Análisis de Overfitting
- Comparación sistemática entre errores de entrenamiento y validación
- Detección de memorización en modelos complejos
- Priorización de capacidad de generalización

## 🏆 Logros del Proyecto

- ✅ **Accuracy superior al objetivo**: 94.24% vs objetivo de >90%
- ✅ **Dataset balanceado**: Perfect 50-50 entre estrellas y galaxias
- ✅ **Pipeline robusto**: Manejo de datos astronómicos reales
- ✅ **Escalabilidad**: Procesamiento de millones de observaciones
- ✅ **Metodología sólida**: Validación cruzada estratificada

## 🚧 Desafíos Enfrentados

### 1. **Tamaño Masivo del Dataset**
- **Problema**: 4 millones de observaciones
- **Solución**: Muestreo estratificado para desarrollo

### 2. **Alta Dimensionalidad**
- **Problema**: 51 variables originales
- **Solución**: Selección basada en correlaciones y conocimiento del dominio

### 3. **Overfitting en Modelos Complejos**
- **Problema**: Random Forest con memorización completa
- **Solución**: Análisis comparativo de errores y selección de modelo estable

### 4. **Distribuciones Sesgadas**
- **Problema**: Variables con distribuciones log-normales extremas
- **Solución**: Transformaciones logarítmicas y escalado robusto

## 💡 Conclusiones Principales

1. **Logistic Regression** demostró ser superior por su capacidad de generalización
2. **Modelos complejos** no garantizan mejor rendimiento en datos reales
3. **Preprocesamiento robusto** es crítico para datos astronómicos
4. **Balance entre eficiencia y precisión** es clave para aplicaciones prácticas

## 🔮 Aplicaciones Futuras

- **Surveys astronómicos** de próxima generación (LSST, Euclid)
- **Clasificación en tiempo real** de alertas astronómicas
- **Extensión a clasificación multi-clase** (diferentes tipos de galaxias)
- **Integración con pipelines** de observatorios automatizados

## 👨‍💻 Autor

**Juan Pablo de Alba Tamayo**
- Proyecto de Maestría en Ciencias de Datos
- Junio 2025

## 📚 Referencias

- **Scikit-learn Documentation**: [https://scikit-learn.org/](https://scikit-learn.org/)
- **SDSS Survey**: [https://www.sdss.org/](https://www.sdss.org/)
- **Kaggle Dataset**: [CelestialClassify](https://www.kaggle.com/datasets/hari31416/celestialclassify)

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

---

⭐ **Si este proyecto te resultó útil, ¡no olvides darle una estrella!** ⭐ 