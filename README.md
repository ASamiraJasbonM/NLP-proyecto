# JustIA — Clasificación Automática de Textos Jurídicos

Sistema de clasificación de consultas legales en 5 categorías (Familia, Laboral, Penal, Civil, Administrativo) usando TF-IDF + Naive Bayes. Proyecto académico para la asignatura *Inteligencia Artificial para el Desarrollo de Software* — Asturias Corporación Universitaria.

## Requisitos

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (gestor de entornos y paquetes)

## Instalación

```bash
uv venv
uv pip install -r requirements.txt
```

## Ejecución

```bash
uv run python -m src.main                # modo CLI (entrenar, evaluar, demo)
uv run python -m src.main --web          # interfaz web en http://127.0.0.1:8000
uv run python -m src.main --web --port 8080  # puerto personalizado
```

## Docker

```bash
docker build -t justia .
docker run -p 8000:8000 justia
```

La aplicación entrena el modelo al arrancar y sirve la interfaz web en `http://localhost:8000`.

## Estructura

```
src/
├── __init__.py
├── config.py      # Hiperparámetros y configuraciones (dataclasses)
├── data.py        # Carga de dataset sintético (50 consultas colombianas)
├── model.py       # Clase JustiaClassifier (entrenar, predecir, guardar, cargar)
├── predict.py     # Función clasificar_consulta() con validación
├── evaluate.py    # Métricas (accuracy, classification_report, confusion_matrix)
├── api.py         # FastAPI con crear_app() y UI interactiva
└── main.py        # Entry point (CLI + --web)
```

## Uso como librería

```python
from src.model import JustiaClassifier
from src.data import cargar_datos_sinteticos

clf = JustiaClassifier().entrenar(cargar_datos_sinteticos())
clf.guardar("modelo.pkl")

# Recargar sin re-entrenar
clf2 = JustiaClassifier().cargar("modelo.pkl")
clf2.predecir(["Quiero divorciarme"])
```

## Interfaz Web

`uv run python -m src.main --web` inicia una interfaz dark-theme en http://127.0.0.1:8000 con:

| Ruta | Descripción |
|------|-------------|
| `GET /` | UI interactiva (dataset, matriz de confusión, probador) |
| `POST /predict` | Clasificar una consulta `{"texto": "..."}` |
| `GET /datos` | Dataset completo en JSON |
| `GET /metricas` | Accuracy, reporte y matriz de confusión |

## Dependencias

| Paquete | Versión |
|---------|---------|
| pandas | >= 2.0.0 |
| numpy | >= 1.24.0 |
| scikit-learn | >= 1.3.0 |
| joblib | >= 1.3.0 |
| fastapi | >= 0.100.0 |
| uvicorn | >= 0.20.0 |

## Categorías

| Categoría | Descripción |
|-----------|-------------|
| Familia | Divorcio, custodia, alimentos, ADN |
| Laboral | Despidos, prestaciones, acoso |
| Penal | Hurtos, estafas, lesiones |
| Civil | Arrendamientos, herencias, contratos |
| Administrativo | Tutelas, sanciones, pensiones |

## Aviso

Este sistema es un apoyo a la orientación legal. No reemplaza el criterio de un abogado o juez.