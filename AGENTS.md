# AGENTS.md — JustIA Project Guide

## Project Overview
NLP legal text classifier (TF-IDF + Naive Bayes) with CLI and FastAPI web interface. Categories: Familia, Laboral, Penal, Civil, Administrativo.

## Quick Start
```bash
uv venv && uv pip install -r requirements.txt
uv run python -m src.main                    # CLI mode
uv run python -m src.main --web              # Web UI at http://127.0.0.1:8000
uv run python -m src.main --web --port 8080  # Custom port
```

## Commands
- **Run CLI**: `uv run python -m src.main`
- **Run web**: `uv run python -m src.main --web`
- **Run API only (dev)**: `uv run uvicorn src.api:crear_app --reload` (doesn't work standalone — use --web)
- **Test model inline**: `uv run python -c "from src.model import JustiaClassifier; from src.data import cargar_datos_sinteticos; j = JustiaClassifier().entrenar(cargar_datos_sinteticos()); print(j.predecir(['texto']))"`
- **Get metrics**: `uv run python -c "from src.data import cargar_datos_sinteticos; from src.model import JustiaClassifier; from src.evaluate import evaluar; r = evaluar(JustiaClassifier().entrenar(cargar_datos_sinteticos())); print(r['accuracy'])"`
- **List deps**: `uv pip list`
- **Add dep**: `uv pip install <pkg>` then add to `requirements.txt`
- **No lint/test framework configured** — no pytest, no flake8, no mypy. `src/` has neither tests nor linting config. Do NOT add tests or linting without asking user first.
- **Clean**: `rm -rf .venv __pycache__ src/__pycache__ *.pkl`

## Code Style

### Formatting & Imports
- 4 spaces, no tabs
- Blank line between stdlib/third-party/first-party import groups
- Order: `import` blocks before `from` blocks, alphabetically within each group
- Module-level constants in UPPER_CASE (`TEMPLATES_DIR`, `CASOS_DEMO`)
- Use f-strings for string formatting (not `.format()` or `%`)

### Naming
- `snake_case` for functions, variables, file names
- `PascalCase` for classes, dataclasses, Pydantic models, FastAPI app
- `UPPER_CASE` for module-level constants
- Private methods prefixed with `_` (e.g. `_build_pipeline`)
- Return self from builder methods (`.entrenar()` returns `self` for chaining)

### Types
- Always annotate function signatures (args + return type)
- `Optional[T]` for nullable params (`from typing import Optional`)
- Use `List[T]`, `Tuple[T, ...]`, `dict[K, V]` from typing
- Use `pd.DataFrame` and `pd.Series` directly as types
- Import types at top of file, not inline

### Architecture
- **One class/responsibility per file**: `model.py` → `JustiaClassifier`, `predict.py` → `clasificar_consulta`, `evaluate.py` → `evaluar`
- **Dataclasses for config**: `ModelConfig`, `TrainConfig`, `PathConfig` in `config.py`
- **No print in business logic files** (model, evaluate, predict) — only in main.py CLI output
- **logging** via `log = logging.getLogger(__name__)` with `INFO` level
- `main.py` is the single entry point; `api.py` exports `crear_app()` factory

### API Patterns (FastAPI)
- Use `crear_app(clasificador, dataframe, metricas) -> FastAPI` factory function
- Pydantic models (`BaseModel`) for request/response schemas
- Static files mounted at `/static`
- HTML templates in `templates/` with `{{ placeholder }}` string replacement (not Jinja2)
- Use HTTPException for error responses

### Error Handling
- Custom exception class `ClasificadorError` in `predict.py`
- Guard clauses first: `if not texto.strip(): raise ...`
- `raise ... from e` when wrapping exceptions
- Validate state before operations: `if self.pipeline is None: raise RuntimeError(...)`

### Model Patterns
- `JustiaClassifier` encapsulates sklearn `Pipeline`
- Methods: `.entrenar()`, `.predecir()`, `.predecir_proba()`, `.guardar()`, `.cargar()`, `.obtener_clases()`, `.get_split_shapes()`
- `.guardar()` uses `joblib.dump`; `.cargar()` uses `joblib.load`
- All methods that require a trained pipeline check `self.pipeline is None` first

### Filesystem
- `TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"` — computed from file location
- `.gitignore`: `.venv/`, `__pycache__/`, `*.pyc`, `*.pkl`, `.DS_Store`
- Avoid hardcoded absolute paths

### Warnings
- `warnings.filterwarnings('ignore')` at module level in `main.py` only
