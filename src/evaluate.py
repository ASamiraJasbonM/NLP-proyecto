import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import List, Optional

from src.model import JustiaClassifier


def evaluar(
    clasificador: JustiaClassifier,
) -> dict:
    if clasificador.y_test is None or clasificador.X_test is None:
        raise RuntimeError("No hay datos de prueba. Ejecuta .entrenar() primero.")

    y_pred = clasificador.predecir(list(clasificador.X_test))
    y_true = list(clasificador.y_test)

    acc = accuracy_score(y_true, y_pred)

    reporte = classification_report(y_true, y_pred, output_dict=True)

    labels = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    return {
        "accuracy": round(acc, 4),
        "classification_report": reporte,
        "confusion_matrix": cm_df,
        "labels": labels,
        "y_true": y_true,
        "y_pred": y_pred,
    }
