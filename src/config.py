from dataclasses import dataclass, field
from typing import List


CATEGORIAS: List[str] = field(
    default_factory=lambda: ["Familia", "Laboral", "Penal", "Civil", "Administrativo"]
)


@dataclass
class ModelConfig:
    ngram_range: tuple = (1, 2)
    min_df: int = 1
    max_features: int = 500
    sublinear_tf: bool = True
    alpha: float = 0.5


@dataclass
class TrainConfig:
    test_size: float = 0.3
    random_state: int = 42


@dataclass
class PathConfig:
    model_file: str = "modelo_justia.pkl"
