"""Application configuration for paths, environment, and runtime constants."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable with a safe default fallback."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable with a safe default fallback."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Immutable settings object built once at application startup."""

    PROJECT_ROOT: Path
    DATA_DIR: Path
    DISTRICTS_CSV: Path
    MODEL_PATH: Path
    DB_PATH: Path
    RISK_MAP_HTML: Path
    ENVIRONMENT: str
    DATABASE_URL: str
    SUMMARIZATION_MODEL: str
    SUMMARIZATION_MAX_LENGTH: int
    SUMMARIZATION_MIN_LENGTH: int
    SUMMARIZATION_MIN_WORDS: int
    MODEL_RANDOM_STATE: int
    CLASSIFICATION_ACCURACY_TARGET: float
    RISK_LABELS: tuple[str, ...]
    DISTRICT_FEATURE_COLUMNS: tuple[str, ...]
    DISTRICT_ID_COLUMN: str
    LAT_COLUMN: str
    LON_COLUMN: str
    DEFAULT_MAP_ZOOM: int
    RISK_MARKER_RADIUS: int
    RISK_COLORS: Mapping[str, str]
    API_PREFIX: str
    HISTORY_DEFAULT_LIMIT: int
    HISTORY_MAX_LIMIT: int
    GRADIO_MOUNT_PATH: str
    APP_TITLE: str

    @classmethod
    def from_env(cls) -> AppSettings:
        """Construct settings from environment variables and project defaults."""
        project_root = Path(__file__).resolve().parents[2]
        data_dir = Path(os.getenv("DATA_DIR", project_root / "data"))
        db_path = data_dir / "query_log.db"

        return cls(
            PROJECT_ROOT=project_root,
            DATA_DIR=data_dir,
            DISTRICTS_CSV=data_dir / "districts_data.csv",
            MODEL_PATH=data_dir / "disaster_model.pkl",
            DB_PATH=db_path,
            RISK_MAP_HTML=data_dir / "risk_map.html",
            ENVIRONMENT=os.getenv("ENVIRONMENT", "development"),
            DATABASE_URL=os.getenv("DATABASE_URL", f"sqlite:///{db_path.resolve()}"),
            SUMMARIZATION_MODEL=os.getenv(
                "SUMMARIZATION_MODEL", "facebook/bart-base"
            ),
            SUMMARIZATION_MAX_LENGTH=_env_int("SUMMARIZATION_MAX_LENGTH", 130),
            SUMMARIZATION_MIN_LENGTH=_env_int("SUMMARIZATION_MIN_LENGTH", 30),
            SUMMARIZATION_MIN_WORDS=_env_int("SUMMARIZATION_MIN_WORDS", 12),
            MODEL_RANDOM_STATE=_env_int("MODEL_RANDOM_STATE", 42),
            CLASSIFICATION_ACCURACY_TARGET=_env_float(
                "CLASSIFICATION_ACCURACY_TARGET", 0.70
            ),
            RISK_LABELS=("low", "medium", "high"),
            DISTRICT_FEATURE_COLUMNS=(
                "rainfall_mm",
                "population_density",
                "elevation_m",
                "proximity_river_km",
                "historical_disasters",
            ),
            DISTRICT_ID_COLUMN="district",
            LAT_COLUMN="lat",
            LON_COLUMN="lon",
            DEFAULT_MAP_ZOOM=_env_int("DEFAULT_MAP_ZOOM", 10),
            RISK_MARKER_RADIUS=_env_int("RISK_MARKER_RADIUS", 12),
            RISK_COLORS=MappingProxyType(
                {"low": "green", "medium": "orange", "high": "red"}
            ),
            API_PREFIX="/api",
            HISTORY_DEFAULT_LIMIT=_env_int("HISTORY_DEFAULT_LIMIT", 50),
            HISTORY_MAX_LIMIT=_env_int("HISTORY_MAX_LIMIT", 200),
            GRADIO_MOUNT_PATH=os.getenv("GRADIO_MOUNT_PATH", "/ui"),
            APP_TITLE=os.getenv("APP_TITLE", "Disaster Preparedness MVP"),
        )

    @property
    def IS_DEVELOPMENT(self) -> bool:
        return self.ENVIRONMENT == "development"

    @property
    def IS_PRODUCTION(self) -> bool:
        return self.ENVIRONMENT == "production"

    def ensure_data_dir(self) -> Path:
        """Create the runtime data directory if it does not exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        return self.DATA_DIR


# Singleton application settings object.
settings = AppSettings.from_env()
settings.ensure_data_dir()

# Backward-compatible module aliases.
PROJECT_ROOT = settings.PROJECT_ROOT
DATA_DIR = settings.DATA_DIR
DISTRICTS_CSV = settings.DISTRICTS_CSV
MODEL_PATH = settings.MODEL_PATH
DB_PATH = settings.DB_PATH
RISK_MAP_HTML = settings.RISK_MAP_HTML
ENVIRONMENT = settings.ENVIRONMENT
IS_DEVELOPMENT = settings.IS_DEVELOPMENT
IS_PRODUCTION = settings.IS_PRODUCTION
DATABASE_URL = settings.DATABASE_URL
SUMMARIZATION_MODEL = settings.SUMMARIZATION_MODEL
SUMMARIZATION_MAX_LENGTH = settings.SUMMARIZATION_MAX_LENGTH
SUMMARIZATION_MIN_LENGTH = settings.SUMMARIZATION_MIN_LENGTH
SUMMARIZATION_MIN_WORDS = settings.SUMMARIZATION_MIN_WORDS
MODEL_RANDOM_STATE = settings.MODEL_RANDOM_STATE
CLASSIFICATION_ACCURACY_TARGET = settings.CLASSIFICATION_ACCURACY_TARGET
RISK_LABELS = settings.RISK_LABELS
DISTRICT_FEATURE_COLUMNS = settings.DISTRICT_FEATURE_COLUMNS
DISTRICT_ID_COLUMN = settings.DISTRICT_ID_COLUMN
LAT_COLUMN = settings.LAT_COLUMN
LON_COLUMN = settings.LON_COLUMN
DEFAULT_MAP_ZOOM = settings.DEFAULT_MAP_ZOOM
RISK_MARKER_RADIUS = settings.RISK_MARKER_RADIUS
RISK_COLORS = settings.RISK_COLORS
API_PREFIX = settings.API_PREFIX
HISTORY_DEFAULT_LIMIT = settings.HISTORY_DEFAULT_LIMIT
HISTORY_MAX_LIMIT = settings.HISTORY_MAX_LIMIT
GRADIO_MOUNT_PATH = settings.GRADIO_MOUNT_PATH
APP_TITLE = settings.APP_TITLE


def ensure_data_dir() -> Path:
    """Backward-compatible function wrapper."""
    return settings.ensure_data_dir()