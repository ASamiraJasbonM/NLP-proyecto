import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, List

from src.config import ModelConfig, TrainConfig, PathConfig, CATEGORIAS


class JustiaClassifier:
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        train_config: Optional[TrainConfig] = None,
    ):
        self.model_config = model_config or ModelConfig()
        self.train_config = train_config or TrainConfig()
        self.pipeline: Optional[Pipeline] = None
        self.X_train: Optional[pd.Series] = None
        self.X_test: Optional[pd.Series] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

    def _build_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=self.model_config.ngram_range,
                        min_df=self.model_config.min_df,
                        max_features=self.model_config.max_features,
                        sublinear_tf=self.model_config.sublinear_tf,
                    ),
                ),
                ("clf", MultinomialNB(alpha=self.model_config.alpha)),
            ]
        )

    def entrenar(self, df: pd.DataFrame) -> "JustiaClassifier":
        X = df["texto"]
        y = df["categoria"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=self.train_config.test_size,
            random_state=self.train_config.random_state,
            stratify=y,
        )

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)
        return self

    def predecir(self, textos: List[str]) -> List[str]:
        if self.pipeline is None:
            raise RuntimeError(
                "El modelo no ha sido entrenado. Llama a .entrenar() primero."
            )
        return list(self.pipeline.predict(textos))

    def predecir_proba(self, textos: List[str]) -> List[dict]:
        if self.pipeline is None:
            raise RuntimeError(
                "El modelo no ha sido entrenado. Llama a .entrenar() primero."
            )
        probas = self.pipeline.predict_proba(textos)
        clases = list(self.pipeline.classes_)
        resultados = []
        for proba in probas:
            resultados.append(
                {clase: round(float(p), 4) for clase, p in zip(clases, proba)}
            )
        return resultados

    def obtener_clases(self) -> List[str]:
        if self.pipeline is None:
            return CATEGORIAS
        return list(self.pipeline.classes_)

    def guardar(self, path_config: Optional[PathConfig] = None) -> str:
        if self.pipeline is None:
            raise RuntimeError("No hay modelo entrenado para guardar.")
        path = (path_config or PathConfig()).model_file
        joblib.dump(self.pipeline, path)
        return path

    def cargar(self, path: str) -> "JustiaClassifier":
        self.pipeline = joblib.load(path)
        return self

    def get_split_shapes(self) -> Tuple[int, int]:
        if self.X_train is None:
            return (0, 0)
        return (len(self.X_train), len(self.X_test))
