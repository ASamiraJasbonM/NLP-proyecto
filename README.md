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
uv run python -m src.main
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
│   └── main.py        # Entry point
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
