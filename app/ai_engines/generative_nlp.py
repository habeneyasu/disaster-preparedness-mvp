"""Generative NLP engine for text summarization and incident dispatch synthesis."""

from __future__ import annotations

import logging
import threading
from typing import Any

from transformers import pipeline

from app.core.config import settings

logger = logging.getLogger(__name__)

_MODEL_LOCK = threading.Lock()
_SUMMARIZATION_PIPELINE: Any | None = None

_FALLBACK_MAX_CHARS = 120
_BART_MAX_INPUT_TOKENS = 1024


def _tokenizer_input_max(tokenizer: Any) -> int:
    """Resolve a safe tokenizer input cap (BART default: 1024 tokens)."""
    model_max = getattr(tokenizer, "model_max_length", _BART_MAX_INPUT_TOKENS)
    if model_max is None or model_max > 10_000:
        model_max = _BART_MAX_INPUT_TOKENS
    return int(model_max)


def _truncate_fallback(text: str, max_chars: int = _FALLBACK_MAX_CHARS) -> str:
    """Return text truncated with ellipsis only when shortening is required."""
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


def _prepare_input_text(summarizer: Any, text: str) -> str:
    """Truncate input text to the model token limit before summarization."""
    tokenizer = summarizer.tokenizer
    model_max = _tokenizer_input_max(tokenizer)
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=model_max,
        return_attention_mask=False,
        add_special_tokens=True,
    )
    return tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)


def _summary_length_bounds(summarizer: Any, text: str) -> tuple[int, int]:
    """Cap output lengths relative to input tokens (BART requires max_length < input)."""
    tokenizer = summarizer.tokenizer
    model_max = _tokenizer_input_max(tokenizer)
    input_length = len(
        tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=model_max,
        )
    )

    max_length = min(
        settings.SUMMARIZATION_MAX_LENGTH,
        max(8, input_length // 2),
    )
    min_length = min(
        settings.SUMMARIZATION_MIN_LENGTH,
        max(5, max_length // 3),
    )
    if min_length >= max_length:
        min_length = max(1, max_length - 1)

    return max_length, min_length


def _get_pipeline() -> Any:
    """Lazy-load and cache the Hugging Face summarization pipeline (thread-safe)."""
    global _SUMMARIZATION_PIPELINE

    if _SUMMARIZATION_PIPELINE is None:
        with _MODEL_LOCK:
            if _SUMMARIZATION_PIPELINE is None:
                logger.info(
                    "Loading summarization model: %s", settings.SUMMARIZATION_MODEL
                )
                _SUMMARIZATION_PIPELINE = pipeline(
                    "summarization",
                    model=settings.SUMMARIZATION_MODEL,
                    device=-1,
                )
                logger.info("Summarization pipeline ready")

    return _SUMMARIZATION_PIPELINE


def generate_report_summary(raw_report: str) -> str:
    """Condense raw field dispatch text into an actionable summary."""
    cleaned_report = raw_report.strip()
    if not cleaned_report:
        return ""

    word_count = len(cleaned_report.split())
    if word_count < settings.SUMMARIZATION_MIN_WORDS:
        logger.debug(
            "Skipping summarization for short report (%s words < %s)",
            word_count,
            settings.SUMMARIZATION_MIN_WORDS,
        )
        return cleaned_report

    try:
        summarizer = _get_pipeline()
        model_input = _prepare_input_text(summarizer, cleaned_report)
        max_length, min_length = _summary_length_bounds(summarizer, model_input)
        logger.debug(
            "Summarizing with max_length=%s, min_length=%s", max_length, min_length
        )

        # Input is pre-truncated; do not pass truncation=True (output max_length differs).
        result = summarizer(
            model_input,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
        )

        if result and isinstance(result, list) and "summary_text" in result[0]:
            summary = str(result[0]["summary_text"]).strip()
            if summary and summary != model_input:
                return summary
            logger.warning(
                "Model returned unchanged text; using truncated original as fallback"
            )
            return _truncate_fallback(cleaned_report)

        raise ValueError("Unexpected summarization pipeline response format")

    except (OSError, RuntimeError, ValueError, IndexError) as err:
        logger.warning(
            "Summarization failed (%s: %s); using truncated original text",
            type(err).__name__,
            err,
        )
        return _truncate_fallback(cleaned_report)
