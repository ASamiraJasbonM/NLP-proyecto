import logging

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.data import cargar_datos_sinteticos
from src.evaluate import evaluar
from src.model import JustiaClassifier
from src.predict import clasificar_consulta, ClasificadorError

log = logging.getLogger(__name__)

app = FastAPI(title="JustIA - Clasificador Jurídico")
clasificador: JustiaClassifier = None
df: pd.DataFrame = None
eval_result: dict = None


class ConsultaInput(BaseModel):
    texto: str


class ConsultaOutput(BaseModel):
    texto: str
    categoria_predicha: str
    confianza: float
    probabilidades: dict
    top_n: list


def inicializar():
    global clasificador, df, eval_result
    df = cargar_datos_sinteticos()
    clasificador = JustiaClassifier().entrenar(df)
    eval_result = evaluar(clasificador)
    clasificador.guardar()
    log.info("Modelo entrenado y API lista")


@app.on_event("startup")
async def startup():
    inicializar()


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>JustIA - Clasificador Jurídico</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0f172a; color: #e2e8f0;
    min-height: 100vh;
  }
  .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
  h1 { font-size: 2rem; font-weight: 700; margin-bottom: .25rem; background: linear-gradient(135deg,#60a5fa,#a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .subtitle { color: #94a3b8; margin-bottom: 2rem; font-size: .95rem; }
  .stats { display: flex; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; }
  .stat-card { background: #1e293b; border-radius: 12px; padding: 1.25rem 1.5rem; flex: 1; min-width: 140px; border: 1px solid #334155; }
  .stat-card .num { font-size: 1.8rem; font-weight: 700; color: #60a5fa; }
  .stat-card .label { color: #94a3b8; font-size: .85rem; }
  .card { background: #1e293b; border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem; border: 1px solid #334155; }
  .card h2 { font-size: 1.15rem; margin-bottom: 1rem; color: #f1f5f9; }
  textarea {
    width: 100%; padding: .85rem; border-radius: 8px; border: 1px solid #334155;
    background: #0f172a; color: #e2e8f0; font-size: .95rem; resize: vertical;
    font-family: inherit; margin-bottom: 1rem;
  }
  textarea:focus { outline: none; border-color: #60a5fa; }
  .btn {
    background: linear-gradient(135deg,#3b82f6,#8b5cf6); color: #fff; border: none;
    padding: .7rem 2rem; border-radius: 8px; font-size: .95rem; font-weight: 600;
    cursor: pointer; transition: opacity .2s;
  }
  .btn:hover { opacity: .85; }
  .btn:disabled { opacity: .5; cursor: not-allowed; }
  #result { margin-top: 1rem; padding: 1rem; border-radius: 8px; display: none; }
  #result.show { display: block; }
  .pred { background: #065f46; border: 1px solid #10b981; }
  .error { background: #7f1d1d; border: 1px solid #ef4444; }
  .prob-bar { display: flex; height: 24px; border-radius: 6px; overflow: hidden; margin-top: .75rem; }
  .prob-seg { display: flex; align-items: center; justify-content: center; font-size: .7rem; font-weight: 600; color: #fff; transition: width .4s; min-width: fit-content; padding: 0 4px; }
  table { width: 100%; border-collapse: collapse; font-size: .85rem; }
  th, td { padding: .6rem .75rem; text-align: left; border-bottom: 1px solid #334155; }
  th { color: #94a3b8; font-weight: 600; position: sticky; top: 0; background: #1e293b; }
  .badge { display: inline-block; padding: .2rem .7rem; border-radius: 20px; font-size: .75rem; font-weight: 600; background: #334155; color: #e2e8f0; }
  .scroll { max-height: 400px; overflow-y: auto; border-radius: 8px; }
  .cm-table { width: auto; margin: 0 auto; }
  .cm-table th { background: #334155; }
  .loader { display: inline-block; width: 16px; height: 16px; border: 2px solid #94a3b8; border-top-color: #60a5fa; border-radius: 50%; animation: spin .6s linear infinite; vertical-align: middle; margin-right: 6px; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="container">
  <h1>&#x2696;&#xFE0F; JustIA</h1>
  <p class="subtitle">Clasificaci&oacute;n Autom&aacute;tica de Textos Jur&iacute;dicos</p>

  <div class="stats">
    <div class="stat-card"><div class="num">__REGISTROS__</div><div class="label">Registros</div></div>
    <div class="stat-card"><div class="num">__CATEGORIAS__</div><div class="label">Categor&iacute;as</div></div>
    <div class="stat-card"><div class="num">__ACCURACY__</div><div class="label">Accuracy</div></div>
  </div>

  <div class="card">
    <h2>&#x1F9EA; Probar clasificador</h2>
    <textarea id="inputText" rows="3" placeholder="Escribe una consulta jur&iacute;dica...">Mi jefe no me paga el subsidio de transporte y tampoco las cesant&iacute;as.</textarea>
    <button class="btn" id="predictBtn" onclick="predecir()">Clasificar</button>
    <div id="result"></div>
  </div>

  <div class="card">
    <h2>&#x1F4CA; Matriz de Confusi&oacute;n</h2>
    __MATRIZ__
  </div>

  <div class="card">
    <h2>&#x1F4C2; Dataset (__REGISTROS__ registros)</h2>
    <div class="scroll">
      <table><thead><tr><th>Texto</th><th>Categor&iacute;a</th></tr></thead><tbody>
        __FILAS__
      </tbody></table>
    </div>
  </div>
</div>

<script>
async function predecir() {
  const btn = document.getElementById('predictBtn');
  const result = document.getElementById('result');
  const texto = document.getElementById('inputText').value.trim();
  if (!texto) { result.className = 'error show'; result.innerHTML = '&#x26A0;&#xFE0F; Ingresa un texto.'; return; }
  btn.disabled = true; btn.innerHTML = '<span class="loader"></span> Clasificando...';
  result.className = '';
  try {
    const res = await fetch('/predict', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({texto}) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Error');
    let bars = '';
    const colors = ['#3b82f6','#8b5cf6','#ef4444','#10b981','#f59e0b'];
    const total = data.top_n.reduce((a,b) => a + b[1], 0);
    data.top_n.forEach((p, i) => {
      const pct = ((p[1] / total) * 100).toFixed(1);
      bars += '<div class="prob-seg" style="width:' + pct + '%;background:' + colors[i % colors.length] + '">' + p[0] + ' ' + pct + '%</div>';
    });
    result.className = 'pred show';
    result.innerHTML = '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.5rem">' +
      '<strong>Categor&iacute;a: <span style="color:#34d399">' + data.categoria_predicha + '</span></strong>' +
      '<span>Confianza: <strong>' + data.confianza + '%</strong></span></div>' +
      '<div class="prob-bar">' + bars + '</div>';
  } catch(e) {
    result.className = 'error show';
    result.innerHTML = '&#x26A0;&#xFE0F; ' + e.message;
  } finally {
    btn.disabled = false; btn.innerHTML = 'Clasificar';
  }
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    categorias = sorted(df["categoria"].unique())
    acc_pct = f"{eval_result['accuracy']:.0%}"

    filas = "".join(
        f"<tr><td>{r['texto']}</td><td><span class='badge'>{r['categoria']}</span></td></tr>"
        for _, r in df.iterrows()
    )

    html = HTML_TEMPLATE
    html = html.replace("__REGISTROS__", str(len(df)))
    html = html.replace("__CATEGORIAS__", str(len(categorias)))
    html = html.replace("__ACCURACY__", acc_pct)
    html = html.replace("__FILAS__", filas)
    html = html.replace(
        "__MATRIZ__", eval_result["confusion_matrix"].to_html(classes="cm-table")
    )

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
    return df.to_dict(orient="records")


@app.get("/metricas")
async def get_metricas():
    return {
        "accuracy": eval_result["accuracy"],
        "classification_report": eval_result["classification_report"],
        "confusion_matrix": eval_result["confusion_matrix"].to_dict(),
    }
