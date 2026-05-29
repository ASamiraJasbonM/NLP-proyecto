import logging
import warnings

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


def main() -> None:
    imprimir_encabezado("JustIA - Clasificación Automática de Textos Jurídicos")

    model_config = ModelConfig()
    train_config = TrainConfig()
    clasificador = JustiaClassifier(
        model_config=model_config, train_config=train_config
    )

    imprimir_encabezado("FASE 1: Datos y Entrenamiento")
    fase_datos(clasificador)

    imprimir_encabezado("FASE 2: Evaluación")
    fase_evaluacion(clasificador)

    imprimir_encabezado("FASE 3: Demostración")
    fase_demo(clasificador)

    ruta = clasificador.guardar()
    log.info("Modelo guardado en: %s", ruta)

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


if __name__ == "__main__":
    main()
