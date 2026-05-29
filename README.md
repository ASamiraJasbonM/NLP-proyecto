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
uv run python -m src.main              # modo CLI (entrenar, evaluar, demo)
uv run python -m src.main --web        # interfaz web FastAPI en http://127.0.0.1:8000
```

## Estructura

```
.
├── src/
│   ├── __init__.py
│   ├── config.py      # Hiperparámetros y configuraciones
│   ├── data.py        # Carga de dataset sintético
│   ├── model.py       # Pipeline TF-IDF + Naive Bayes
│   ├── predict.py     # Función de inferencia
│   ├── evaluate.py    # Métricas de evaluación
│   ├── api.py         # FastAPI (endpoints /, /predict, /datos, /metricas)
│   └── main.py        # Entry point (CLI + --web)
├── requirements.txt
├── justia_mvp.py      # Versión original (monolítica)
└── README.md
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

La API FastAPI ofrece:

| Ruta | Descripción |
|------|-------------|
| `GET /` | UI interactiva (dataset, matriz de confusión, probador) |
| `POST /predict` | Clasificar una consulta (JSON) |
| `GET /datos` | Dataset completo en JSON |
| `GET /metricas` | Accuracy, reporte y matriz de confusión |

## Categorías

| Categoría       | Descripción                          |
|-----------------|--------------------------------------|
| Familia         | Divorcio, custodia, alimentos, ADN   |
| Laboral         | Despidos, prestaciones, acoso        |
| Penal           | Hurtos, estafas, lesiones            |
| Civil           | Arrendamientos, herencias, contratos |
| Administrativo  | Tutelas, sanciones, pensiones        |

## Aviso

Este sistema es un apoyo a la orientación legal. No reemplaza el criterio de un abogado o juez.
