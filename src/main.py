import argparse
import logging
import sys
import warnings

import pandas as pd

from src.config import ModelConfig, TrainConfig
from src.data import cargar_datos_sinteticos
from src.model import JustiaClassifier
from src.evaluate import evaluar
from src.predict import clasificar_consulta, ClasificadorError

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


CASOS_DEMO = [
    "Mi jefe no me paga el subsidio de transporte y tampoco las cesantías.",
    "Fui víctima de violación de domicilio y quiero denunciarlo.",
    "Necesito disolver la sociedad limitada que tengo con mi hermano.",
    "La DIAN me embargó la cuenta sin notificarme previamente.",
    "Quiero recuperar la custodia de mi hija después de la separación.",
]


def imprimir_encabezado(titulo: str) -> None:
    print()
    print("=" * 70)
    print(f"  {titulo}")
    print("=" * 70)


def fase_datos(clasificador: JustiaClassifier) -> None:
    df = cargar_datos_sinteticos()
    print(f"Total de registros: {len(df)}")
    print(f"Categorías: {df['categoria'].unique().tolist()}")
    print(f"\nDistribución por categoría:")
    print(df["categoria"].value_counts().to_string())

    clasificador.entrenar(df)
    n_train, n_test = clasificador.get_split_shapes()
    print(f"\nDivisión: {n_train} entrenamiento, {n_test} prueba")
    print("Modelo entrenado exitosamente.")


def fase_evaluacion(clasificador: JustiaClassifier) -> None:
    resultado = evaluar(clasificador)
    print(f"\nAccuracy: {resultado['accuracy']:.2%}")
    print("\nMatriz de Confusión:")
    print(resultado["confusion_matrix"].to_string())


def fase_demo(clasificador: JustiaClassifier) -> None:
    for caso in CASOS_DEMO:
        try:
            res = clasificar_consulta(caso, clasificador, top_n=2)
            texto = res["texto"]
            if len(texto) > 70:
                texto = texto[:70] + "..."
            print(f'Consulta: "{texto}"')
            print(
                f"  -> Categoría: {res['categoria_predicha']} (Confianza: {res['confianza']}%)"
            )
            for cat, prob in res["top_n"]:
                print(f"     {cat}: {prob:.4f}")
            print()
        except ClasificadorError as e:
            log.error("Error clasificando '%s': %s", caso, e)


def main(clasificador: JustiaClassifier) -> None:
    imprimir_encabezado("JustIA - Clasificación Automática de Textos Jurídicos")

    imprimir_encabezado("FASE 1: Datos y Entrenamiento")
    n_train, n_test = clasificador.get_split_shapes()
    print(f"Registros: {n_train + n_test}")
    print(f"Categorías: {clasificador.obtener_clases()}")
    print(f"División: {n_train} entrenamiento, {n_test} prueba")
    print("Modelo entrenado exitosamente.")

    imprimir_encabezado("FASE 2: Evaluación")
    fase_evaluacion(clasificador)

    imprimir_encabezado("FASE 3: Demostración")
    fase_demo(clasificador)

    print()
    print("=" * 70)
    print("  RESUMEN")
    print("=" * 70)
    print(f"""
  Modelo:       TF-IDF + Multinomial Naive Bayes
  Pipeline:     scikit-learn
  Categorías:   Familia, Laboral, Penal, Civil, Administrativo
  Dataset:      50 muestras sintéticas (10 por categoría)
  Uso:          from src.model import JustiaClassifier
                clf = JustiaClassifier().cargar("modelo_justia.pkl")
                clf.predecir(["texto legal"])
    """)
    print("=" * 70)


def iniciar_web(
    clasificador: JustiaClassifier,
    dataframe: pd.DataFrame,
    metricas: dict,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    import uvicorn
    from src.api import crear_app

    app = crear_app(clasificador, dataframe, metricas)
    log.info("Interfaz web en http://%s:%s", host, port)
    uvicorn.run(app, host=host, port=port)


def entrenar_y_evaluar():
    imprimir_encabezado("JustIA - Clasificación Automática de Textos Jurídicos")

    model_config = ModelConfig()
    train_config = TrainConfig()
    clasificador = JustiaClassifier(
        model_config=model_config, train_config=train_config
    )

    df = cargar_datos_sinteticos()
    clasificador.entrenar(df)
    metricas = evaluar(clasificador)
    clasificador.guardar()

    log.info("Modelo guardado. Accuracy: %.2f%%", metricas["accuracy"] * 100)
    return clasificador, df, metricas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JustIA - Clasificador Jurídico")
    parser.add_argument(
        "--web", action="store_true", help="Iniciar interfaz web FastAPI"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host para el servidor web")
    parser.add_argument(
        "--port", type=int, default=8000, help="Puerto para el servidor web"
    )
    args = parser.parse_args()

    clf, df, metricas = entrenar_y_evaluar()

    if args.web:
        iniciar_web(clf, df, metricas, host=args.host, port=args.port)
    else:
        main(clf)
