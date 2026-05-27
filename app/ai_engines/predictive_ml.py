"""Predictive ML: district features + summary → risk label and confidence."""

from __future__ import annotations

import logging
import pickle
import threading
from dataclasses import dataclass
from typing import Any

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from app.core.config import settings

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_BUNDLE: ModelBundle | None = None
_DISTRICTS: pd.DataFrame | None = None

_LABEL_PHRASES = {
    "low": "stable conditions minimal impact",
    "medium": "moderate impact ongoing response",
    "high": "severe flooding critical evacuation",
}


@dataclass
class ModelBundle:
    classifier: RandomForestClassifier
    vectorizer: TfidfVectorizer
    scaler: StandardScaler
    training_accuracy: float | None = None


def _districts() -> pd.DataFrame:
    global _DISTRICTS
    if _DISTRICTS is not None:
        return _DISTRICTS
    if not settings.DISTRICTS_CSV.exists():
        raise FileNotFoundError(f"District seed data not found: {settings.DISTRICTS_CSV}")
    df = pd.read_csv(settings.DISTRICTS_CSV)
    required = {settings.DISTRICT_ID_COLUMN, *settings.DISTRICT_FEATURE_COLUMNS}
    if missing := required - set(df.columns):
        raise ValueError(f"districts_data.csv missing columns: {sorted(missing)}")
    _DISTRICTS = df
    return df


def _district_row(district: str) -> pd.Series:
    df = _districts()
    matches = df[df[settings.DISTRICT_ID_COLUMN] == district]
    if matches.empty:
        raise ValueError(f"Unknown district: {district}")
    return matches.iloc[0]


def _derive_label(row: pd.Series) -> str:
    score = (
        float(row["historical_disasters"]) * 2.0
        + float(row["rainfall_mm"]) / 120.0
        + 1.0 / (float(row["proximity_river_km"]) + 0.2)
    )
    if score >= 14.0:
        return "high"
    if score >= 7.0:
        return "medium"
    return "low"


def _feature_text(row: pd.Series, label: str | None = None) -> str:
    parts = " ".join(f"{c} {row[c]}" for c in settings.DISTRICT_FEATURE_COLUMNS)
    if label:
        return f"{parts} {_LABEL_PHRASES[label]}"
    return parts


def _features(
    df: pd.DataFrame,
    texts: list[str],
    scaler: StandardScaler | None = None,
    vectorizer: TfidfVectorizer | None = None,
    *,
    fit: bool = False,
) -> tuple[Any, StandardScaler, TfidfVectorizer]:
    numeric = df[list(settings.DISTRICT_FEATURE_COLUMNS)].to_numpy(dtype=float)
    if fit:
        scaler = StandardScaler()
        vectorizer = TfidfVectorizer(max_features=256, ngram_range=(1, 2))
        x_num = csr_matrix(scaler.fit_transform(numeric))
        x_txt = vectorizer.fit_transform(texts)
    else:
        if scaler is None or vectorizer is None:
            raise ValueError("scaler and vectorizer required")
        x_num = csr_matrix(scaler.transform(numeric))
        x_txt = vectorizer.transform(texts)
    return hstack([x_num, x_txt]), scaler, vectorizer


def _train() -> ModelBundle:
    df = _districts()
    labels = [_derive_label(row) for _, row in df.iterrows()]
    texts = [_feature_text(row, label) for (_, row), label in zip(df.iterrows(), labels)]

    x, scaler, vectorizer = _features(df, texts, fit=True)
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=settings.MODEL_RANDOM_STATE,
        class_weight="balanced",
    )
    clf.fit(x, labels)

    accuracy = accuracy_score(labels, clf.predict(x))
    logger.info("Model training accuracy: %.2f", accuracy)
    if accuracy < settings.CLASSIFICATION_ACCURACY_TARGET:
        logger.warning(
            "Accuracy %.2f below target %.2f",
            accuracy,
            settings.CLASSIFICATION_ACCURACY_TARGET,
        )

    bundle = ModelBundle(
        classifier=clf,
        vectorizer=vectorizer,
        scaler=scaler,
        training_accuracy=float(accuracy),
    )
    settings.ensure_data_dir()
    with settings.MODEL_PATH.open("wb") as f:
        pickle.dump(bundle, f)
    return bundle


def _get_bundle() -> ModelBundle:
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE
    with _LOCK:
        if _BUNDLE is not None:
            return _BUNDLE
        if settings.MODEL_PATH.exists():
            with settings.MODEL_PATH.open("rb") as f:
                loaded = pickle.load(f)
            if isinstance(loaded, ModelBundle):
                _BUNDLE = loaded
                return _BUNDLE
        _BUNDLE = _train()
        return _BUNDLE


def _training_accuracy(bundle: ModelBundle) -> float:
    stored = getattr(bundle, "training_accuracy", None)
    if stored is not None:
        return float(stored)
    df = _districts()
    labels = [_derive_label(row) for _, row in df.iterrows()]
    texts = [_feature_text(row, label) for (_, row), label in zip(df.iterrows(), labels)]
    x, _, _ = _features(df, texts, bundle.scaler, bundle.vectorizer, fit=False)
    return float(accuracy_score(labels, bundle.classifier.predict(x)))


def get_classification_benchmark() -> tuple[float, float]:
    """Return (model training accuracy, configured benchmark target)."""
    bundle = _get_bundle()
    return _training_accuracy(bundle), settings.CLASSIFICATION_ACCURACY_TARGET


def predict_risk(district: str, summary: str) -> tuple[str, float]:
    """Return (predicted_risk, confidence) for a district and NLP summary."""
    bundle = _get_bundle()
    row = _district_row(district)
    text = summary.strip() or _feature_text(row, _derive_label(row))

    x, _, _ = _features(
        pd.DataFrame([row]), [text], bundle.scaler, bundle.vectorizer, fit=False
    )
    risk = str(bundle.classifier.predict(x)[0])
    proba = bundle.classifier.predict_proba(x)[0]
    conf = float(proba[list(bundle.classifier.classes_).index(risk)])

    if risk not in settings.RISK_LABELS:
        return "medium", round(conf, 4)
    return risk, round(conf, 4)


def get_district_coordinates(district: str) -> tuple[float, float]:
    row = _district_row(district)
    return float(row[settings.LAT_COLUMN]), float(row[settings.LON_COLUMN])


def list_districts() -> list[str]:
    df = _districts()
    return df[settings.DISTRICT_ID_COLUMN].tolist()
