from typing import Optional

from src.model import JustiaClassifier


class ClasificadorError(Exception):
    """Error relacionado con la clasificación de textos."""


def clasificar_consulta(
    texto: str,
    clasificador: JustiaClassifier,
    top_n: Optional[int] = None,
) -> dict:
    if not texto or not texto.strip():
        raise ClasificadorError("El texto de la consulta no puede estar vacío.")

    try:
        categoria = clasificador.predecir([texto])[0]
        probas = clasificador.predecir_proba([texto])[0]
    except Exception as e:
        raise ClasificadorError(f"Error al clasificar el texto: {e}") from e

    resultado = {
        "texto": texto,
        "categoria_predicha": categoria,
        "confianza": round(max(probas.values()) * 100, 1),
        "probabilidades": probas,
    }

    if top_n and top_n > 0:
        resultado["top_n"] = sorted(probas.items(), key=lambda x: x[1], reverse=True)[
            :top_n
        ]

    return resultado
