# Alerta temprana de cambio de carrera — Región del Biobío

**ACIF104 Aprendizaje de Máquina · Universidad Andrés Bello · Sumativa 1 · Grupo 2**

Integrantes: Christian Fuentes, Bastián González, Javier Jorquera y Andrea Luna

---

## 1. Descripción del proyecto

Las instituciones de educación superior necesitan identificar a tiempo a los estudiantes con mayor probabilidad de cambiar de carrera, para orientar acciones de acompañamiento y retención. Este proyecto construye un modelo de aprendizaje automático que estima el riesgo de cambio de carrera de un estudiante y lo pone a disposición de usuarios no técnicos mediante una aplicación web con explicación de cada predicción.

- **Datos:** matrícula de educación superior 2025 de la Región del Biobío, publicada por el Ministerio de Educación de Chile (datos abiertos MINEDUC). 136.290 registros y 53 variables.
- **Variable objetivo:** `cambio_carrera` (1 = cambio interno o externo de carrera, 0 = sin cambio). La clase positiva representa aproximadamente el 4 % de los registros.
- **Fuga de información:** se excluyeron `forma_ingreso` (usada para construir el objetivo) y las variables de año y semestre de ingreso a la carrera de origen y actual, que permiten reconstruir directamente el cambio de carrera.

## 2. Resultados principales

| Etapa | Resultado |
|---|---|
| Machine Learning | 3 modelos (Regresión Logística, Random Forest, XGBoost) × 4 estrategias de balanceo (sin balanceo, ROS, SMOTE, RUS) = 12 combinaciones. Mejor: **XGBoost + SMOTE** (F1 validación = 0,3196). |
| Deep Learning | 3 arquitecturas: MLP (F1 = 0,3359), red con embeddings (F1 = 0,3102) y ResNet tabular (F1 = 0,3209), todas con umbral ajustado. |
| Refinamiento | Búsqueda aleatoria de hiperparámetros (RandomizedSearchCV, 3 particiones) y ajuste del umbral de decisión. Versión definitiva: **XGBoost + SMOTE con umbral 0,37** (F1 validación = 0,3456). |
| Evaluación en datos no vistos (prueba) | Precisión = 0,3086 · Recall = 0,2744 · F1 = 0,2905 · ROC AUC = 0,8312 |
| Interpretabilidad | SHAP (TreeSHAP exacto) global, agregado por variable original y para 3 casos particulares (acierto, no detectado y falsa alarma). |
| Aplicación | Backend FastAPI + frontend Streamlit, con explicación SHAP por estudiante, predicción por lotes y monitoreo. |

## 3. Estructura del repositorio

```
acif104_Grupo2_cambio_carrera_biobio/
├── README.md
├── .gitignore
├── notebook/
│   └── acif104_s6_grupo2.ipynb       # Análisis completo: EDA, modelos, SHAP y pruebas de la API
├── data/
│   └── matricula_biobio_dataset.csv  # Dataset utilizado (separador ";")
├── api/
│   ├── api.py                        # Backend (FastAPI)
│   ├── app.py                        # Frontend (Streamlit)
│   ├── modelo_cambio_carrera.joblib  # Preprocesador, umbral, catálogo de programas, etc.
│   ├── modelo_xgboost.json           # Clasificador XGBoost en formato nativo
│   └── requirements.txt              # Dependencias de la aplicación
└── docs/
    ├── acif104_s6_grupo2.pdf     
    └── capturas/                     # Capturas de la aplicación en funcionamiento
```

## 4. Requisitos

- **Python 3.12** (recomendado). Las versiones de las librerías fijadas en `requirements.txt` no tienen instaladores para Python 3.14.
- Windows, macOS o Linux.
- Para el notebook: una cuenta de Google (Google Colab).

## 5. Notebook en Google Colab

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anluni/acif104_Grupo2_cambio_carrera_biobio/blob/main/notebook/acif104_s6_grupo2.ipynb)

### 5.1 Ver el notebook sin ejecutarlo

El notebook se guardó **con todos sus resultados** (tablas, gráficos y métricas), por lo que se puede revisar sin ejecutar nada:

- **En GitHub:** abrir `notebook/acif104_s6_grupo2.ipynb`. GitHub muestra el código y los resultados.
- **En Colab:** presionar el botón **Abrir en Colab** de arriba.
- Si GitHub no logra mostrar el archivo por su tamaño, usar [nbviewer](https://nbviewer.org/github/anluni/acif104_Grupo2_cambio_carrera_biobio/blob/main/notebook/acif104_s6_grupo2.ipynb).

### 5.2 Estructura del notebook

| Sección del notebook | Contenido | Sección del informe |
|---|---|---|
| 1 | Obtención de los datos | 1.4 |
| 2 (2.1 a 2.7) | Análisis exploratorio: descripción de variables, estadísticas, completitud, correctitud, outliers, distribuciones, correlaciones y relación con el cambio de carrera | 3.1 |
| 3 | Preparación de los datos, variable objetivo y exclusión de variables con fuga de información | 3.4.1 |
| 4 y 5 | Partición estratificada (70/15/15) y preprocesamiento | 3.4.1 y 3.4.2 |
| 6 a 10 | Balanceo (ROS, SMOTE, RUS) y modelos de Machine Learning (Regresión Logística, Random Forest, XGBoost) | 3.4.3 y 3.5.1 |
| 11 | Selección del mejor modelo, curva ROC y matriz de confusión | 3.5.1 |
| 13.1 a 13.6 | Arquitecturas de Deep Learning (MLP, red con embeddings, ResNet tabular) y comparación con ML | 3.5.2 y 3.5.3 |
| 13.7 | Refinamiento: hiperparámetros, ajuste del umbral y evaluación en prueba | 3.6 y 4.1 |
| 14 | Interpretabilidad con SHAP (global y casos particulares) | 4.2 y 4.3 |
| 15 | Exportación del modelo, backend (FastAPI), frontend (Streamlit) y pruebas de la API | 3.7 |
| 16 | Conclusiones finales | 7 |

### 5.3 Ejecutar el notebook

1. Presionar **Abrir en Colab** (o en Colab: **Archivo → Subir notebook**).
2. En el panel de archivos de Colab (ícono de carpeta, a la izquierda), subir `data/matricula_biobio_dataset.csv` y **esperar a que la carga termine** antes de continuar. El archivo completo tiene 136.291 líneas, incluido el encabezado, y se puede verificar con `!wc -l matricula_biobio_dataset.csv`.
3. Ejecutar todo: **Entorno de ejecución → Ejecutar todas**. La ejecución completa tarda entre 30 y 40 minutos.
4. Al finalizar, la sección 15 genera la carpeta `app_cambio_carrera/` con los archivos de la aplicación.

## 6. Instalación y ejecución de la aplicación

### 6.1 Instalación (solo la primera vez)

Abrir una terminal dentro de la carpeta `api/`:

**Windows (PowerShell):**
```powershell
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Si PowerShell bloquea la activación del ambiente, ejecutar antes:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

**macOS / Linux:**
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 6.2 Ejecución

Se necesitan **dos terminales**, ambas dentro de `api/` y con el ambiente activado.

**Terminal 1: backend**
```bash
uvicorn api:app --reload
```
- API: http://localhost:8000
- Documentación interactiva: http://localhost:8000/docs

**Terminal 2: frontend**
```bash
streamlit run app.py
```
- Aplicación web: http://localhost:8501

Para detener la aplicación, presionar `Ctrl + C` en cada terminal.

### 6.3 Uso

1. En la pestaña **Evaluar estudiante**, seleccionar institución, sede, carrera, jornada y modalidad, junto con el género y la fecha de nacimiento del estudiante. El rango de edad se completa automáticamente.
2. Presionar **Evaluar riesgo**. La aplicación muestra el puntaje de riesgo, la alerta (si el puntaje es mayor o igual a 0,37) y las cinco variables que más influyen en la predicción según SHAP.
3. En la pestaña **Monitoreo**, revisar el número de predicciones, la tasa de alertas comparada con la esperada (3,6 %), el tiempo de respuesta y el historial de consultas.

> El puntaje es un **indicador de riesgo relativo**, no una probabilidad calibrada, porque el modelo fue entrenado con datos balanceados mediante SMOTE.

## 7. API (backend)

| Endpoint | Método | Descripción |
|---|---|---|
| `/salud` | GET | Estado del servicio, modelo y umbral. |
| `/programas` | GET | Catálogo de los 1.640 programas de estudio. |
| `/fechas` | GET | Fechas de nacimiento válidas. |
| `/predecir` | POST | Puntaje, alerta y explicación SHAP de un estudiante. |
| `/predecir_lote` | POST | Predicción para una lista de estudiantes. |
| `/monitoreo` | GET | Indicadores de uso y posible desviación de los datos. |

Ejemplo de solicitud a `/predecir`:
```json
{
  "codigo_unico": "I456S4C42J2V1",
  "gen_alu": "Femenino",
  "fec_nac_alu": "1983-04-01"
}
```

Ejemplo de respuesta:
```json
{
  "puntaje_riesgo": 0.0403,
  "umbral": 0.37,
  "alerta": false,
  "mensaje": "Riesgo bajo: no se requiere acción inmediata.",
  "rango_edad": "35 a 39 años",
  "factores": [
    {"variable": "fec_nac_alu", "contribucion": -1.3353, "efecto": "disminuye el riesgo"}
  ],
  "latencia_ms": 33.2
}
```

Cada predicción se registra en `registro_predicciones.csv` (fecha y hora, programa, datos de entrada, puntaje, alerta y latencia), que alimenta el monitoreo.

## 8. Limitaciones y propuestas de mejora

- El modelo se apoya principalmente en características del programa y de la institución. Incorporar variables académicas y socioeconómicas (rendimiento, asistencia, puntaje de ingreso, beneficios) permitiría detectar cambios de carrera en estudiantes con perfiles tradicionales.
- Transformar `fec_nac_alu` en una variable numérica de edad y tratar `cod_inst` y `cod_sede` como variables categóricas.
- Eliminar variables sin variación o redundantes (`region_sede`, `formato_valores`, `dur_total_carr`).
- Optimizar hiperparámetros con más iteraciones o con optimización bayesiana, y calibrar las probabilidades del modelo.
- Reemplazar la regla simple de detección de desviaciones por pruebas estadísticas de drift (PSI o Kolmogorov-Smirnov).
- Desplegar el backend y el frontend en un servidor o en contenedores (Docker) para su uso institucional.
