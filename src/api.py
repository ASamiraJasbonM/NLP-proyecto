import logging
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.model import JustiaClassifier
from src.predict import clasificar_consulta, ClasificadorError

log = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


class ConsultaInput(BaseModel):
    texto: str


class ConsultaOutput(BaseModel):
    texto: str
    categoria_predicha: str
    confianza: float
    probabilidades: dict
    top_n: list


def crear_app(
    clasificador: JustiaClassifier,
    dataframe: pd.DataFrame,
    metricas: dict,
) -> FastAPI:
    app = FastAPI(title="JustIA - Clasificador Jurídico")
    app.mount("/static", StaticFiles(directory=str(TEMPLATES_DIR)), name="static")

    template_path = TEMPLATES_DIR / "index.html"

    @app.get("/", response_class=HTMLResponse)
    async def index():
        categorias = sorted(dataframe["categoria"].unique())
        acc_pct = f"{metricas['accuracy']:.0%}"

        filas = "".join(
            f"<tr><td>{r['texto']}</td><td><span class='badge'>{r['categoria']}</span></td></tr>"
            for _, r in dataframe.iterrows()
        )

        reporte = metricas["classification_report"]
        labels = metricas["labels"]

        accuracy = metricas["accuracy"]
        metricas_rows = ""
        for label in labels:
            m = reporte[label]
            metricas_rows += (
                f"<tr><td>{label}</td>"
                f"<td>{m['precision']:.2f}</td>"
                f"<td>{m['recall']:.2f}</td>"
                f"<td>{m['f1-score']:.2f}</td>"
                f"<td>{int(m['support'])}</td></tr>"
            )
        metricas_rows += (
            f"<tr class='avg-row'><td>Promedio macro</td>"
            f"<td>{reporte['macro avg']['precision']:.2f}</td>"
            f"<td>{reporte['macro avg']['recall']:.2f}</td>"
            f"<td>{reporte['macro avg']['f1-score']:.2f}</td>"
            f"<td>{int(reporte['macro avg']['support'])}</td></tr>"
        )
        metricas_rows += (
            f"<tr class='avg-row'><td>Promedio ponderado</td>"
            f"<td>{reporte['weighted avg']['precision']:.2f}</td>"
            f"<td>{reporte['weighted avg']['recall']:.2f}</td>"
            f"<td>{reporte['weighted avg']['f1-score']:.2f}</td>"
            f"<td>{int(reporte['weighted avg']['support'])}</td></tr>"
        )
        metricas_rows += (
            f"<tr class='accuracy-row'><td>Exactitud (Accuracy)</td>"
            f"<td colspan='4'>{accuracy:.2%}</td></tr>"
        )

        html = template_path.read_text(encoding="utf-8")
        html = html.replace("{{ registros }}", str(len(dataframe)))
        html = html.replace("{{ categorias }}", str(len(categorias)))
        html = html.replace("{{ accuracy }}", acc_pct)
        html = html.replace("{{ filas }}", filas)
        html = html.replace(
            "{{ matriz }}", metricas["confusion_matrix"].to_html(classes="cm-table")
        )
        html = html.replace("{{ metricas_rows }}", metricas_rows)

        return HTMLResponse(content=html)

    @app.post("/predict", response_model=ConsultaOutput)
    async def predict(input: ConsultaInput):
        if not input.texto.strip():
            raise HTTPException(400, "El texto no puede estar vacío")
        try:
            res = clasificar_consulta(input.texto, clasificador, top_n=5)
            return ConsultaOutput(
                texto=res["texto"],
                categoria_predicha=res["categoria_predicha"],
                confianza=res["confianza"],
                probabilidades=res["probabilidades"],
                top_n=res["top_n"],
            )
        except ClasificadorError as e:
            raise HTTPException(400, str(e))

    @app.get("/datos")
    async def get_datos():
        return dataframe.to_dict(orient="records")

    @app.get("/metricas")
    async def get_metricas():
        return {
            "accuracy": metricas["accuracy"],
            "classification_report": metricas["classification_report"],
            "confusion_matrix": metricas["confusion_matrix"].to_dict(),
        }

    return app
