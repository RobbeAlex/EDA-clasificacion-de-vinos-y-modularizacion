# EDA, Clasificación de Vinos y Modularización

Pipeline modular de Machine Learning en Python para el análisis exploratorio de datos (EDA), entrenamiento supervisado multiclase y evaluación de modelos predictivos sobre la calidad del vino tinto (*Red Wine Quality Dataset*).

---

## Objetivo del Proyecto

Estructurar un flujo de trabajo reproducible y modular en ingeniería de Machine Learning separando la exploración estadística, el ajuste y selección de clasificadores, y la validación final sobre datos de prueba, prediciendo la puntuación sensorial de calidad (`quality`) a partir de 11 variables físico-químicas.

---

## Conjunto de Datos (`data/winequality-red.csv`)

El dataset contiene registros de variantes del vino tinto portugués *"Vinho Verde"* con las siguientes características de entrada:
- `fixed acidity`: Acidez fija ($\text{g/dm}^3$)
- `volatile acidity`: Acidez volátil ($\text{g/dm}^3$)
- `citric acid`: Ácido cítrico ($\text{g/dm}^3$)
- `residual sugar`: Azúcar residual ($\text{g/dm}^3$)
- `chlorides`: Cloruros ($\text{g/dm}^3$)
- `free sulfur dioxide`: Dióxido de azufre libre ($\text{mg/dm}^3$)
- `total sulfur dioxide`: Dióxido de azufre total ($\text{mg/dm}^3$)
- `density`: Densidad ($\text{g/cm}^3$)
- `pH`: Nivel de pH
- `sulphates`: Sulfatos ($\text{g/dm}^3$)
- `alcohol`: Porcentaje de alcohol ($\%\text{ vol.}$)
- **`quality`** *(Variable objetivo)*: Calificación sensorial discreta entre 3 y 8.

---

## Estructura del Proyecto

```text
EDA, clasificacion de vinos y modularizacion/
├── data/
│   └── winequality-red.csv       # Dataset de entrada (propiedades físico-químicas)
├── models/
│   └── mejor_modelo.pkl          # Artefacto serializado del mejor clasificador
├── src/
│   ├── eda.py                    # Análisis exploratorio y visualizaciones estadísticas
│   ├── entrenamiento.py          # Entrenamiento comparativo (Random Forest vs SVM) y persistencia
│   └── test/prueba.py                 # Evaluación del modelo serializado y matriz de confusión
├── DOCUMENTACION_TECNICA.md      # Arquitectura técnica, flujo de datos y enlaces a código
├── LICENSE                       # Licencia MIT
├── README.md                     # Documentación general del proyecto
└── requirements.txt              # Dependencias del entorno Python
```

---

## Instalación

1. Crear y activar un entorno virtual (opcional pero recomendado):
   ```bash
   python -m venv venv
   # En Windows:
   venv\Scripts\activate
   ```

2. Instalar las dependencias requeridas:
   ```bash
   pip install -r requirements.txt
   ```

---

## Uso del Pipeline Modular

Los scripts dentro de `src/` resuelven dinámicamente las rutas relativas al directorio raíz del proyecto, por lo que pueden ejecutarse desde cualquier ubicación:

### 1. Análisis Exploratorio de Datos (EDA)
Genera estadísticas descriptivas en consola, mapa de calor de correlación de Pearson, histogramas de distribución y diagramas de caja (*boxplots*) para inspección de valores atípicos:
```bash
python src/eda.py
```

### 2. Entrenamiento y Selección de Modelos
Divide el conjunto de datos (80% entrenamiento / 20% prueba con semilla `random_state=42`), entrena clasificadores `RandomForestClassifier` y `SVC`, compara métricas ponderadas (`Accuracy`, `Precision`, `Recall`, `F1-Score`) y serializa automáticamente el mejor clasificador en `models/mejor_modelo.pkl`:
```bash
python src/entrenamiento.py
```

### 3. Evaluación en Conjunto de Prueba
Carga el artefacto `models/mejor_modelo.pkl`, ejecuta la inferencia sobre la partición de prueba (`X_test`, `y_test`), imprime el reporte de clasificación detallado por clase y despliega la matriz de confusión:
```bash
python test/test/prueba.py
```

---

## Licencia

Distribuido bajo la licencia MIT. Consulta el archivo `LICENSE` para más información.
