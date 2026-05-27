"""Generative NLP: field reports → concise summaries (Hugging Face BART)."""

from __future__ import annotations

import logging
import threading
from typing import Any

from transformers import pipeline

from app.core.config import settings

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_PIPELINE: Any | None = None
_MAX_INPUT_TOKENS = 1024
_FALLBACK_CHARS = 120


def _truncate_fallback(text: str) -> str:
    if len(text) <= _FALLBACK_CHARS:
        return text
    return f"{text[:_FALLBACK_CHARS]}..."


def _tokenizer_max(tokenizer: Any) -> int:
    cap = getattr(tokenizer, "model_max_length", _MAX_INPUT_TOKENS)
    return _MAX_INPUT_TOKENS if cap is None or cap > 10_000 else int(cap)


def _token_count(tokenizer: Any, text: str) -> int:
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=_tokenizer_max(tokenizer),
        )
    )


def _prepare_input(tokenizer: Any, text: str) -> str:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=_tokenizer_max(tokenizer),
        add_special_tokens=True,
    )
    return tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)


def _output_lengths(input_tokens: int) -> tuple[int, int]:
    max_len = min(settings.SUMMARIZATION_MAX_LENGTH, max(8, input_tokens // 2))
    min_len = min(settings.SUMMARIZATION_MIN_LENGTH, max(5, max_len // 3))
    if min_len >= max_len:
        min_len = max(1, max_len - 1)
    return max_len, min_len


def _get_pipeline() -> Any:
    global _PIPELINE
    if _PIPELINE is None:
        with _LOCK:
            if _PIPELINE is None:
                logger.info("Loading %s", settings.SUMMARIZATION_MODEL)
                _PIPELINE = pipeline(
                    "summarization",
                    model=settings.SUMMARIZATION_MODEL,
                    device=-1,
                )
    return _PIPELINE


def generate_report_summary(raw_report: str) -> str:
    """Condense raw field dispatch text into an actionable summary."""
    text = raw_report.strip()
    if not text:
        return ""
    if len(text.split()) < settings.SUMMARIZATION_MIN_WORDS:
        return text

    try:
        pipe = _get_pipeline()
        tokenizer = pipe.tokenizer
        model_input = _prepare_input(tokenizer, text)
        max_len, min_len = _output_lengths(_token_count(tokenizer, model_input))

        result = pipe(
            model_input,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
        )
        summary = str(result[0]["summary_text"]).strip()
        if summary and summary != model_input:
            return summary
        return _truncate_fallback(text)
    except (OSError, RuntimeError, ValueError, IndexError) as err:
        logger.warning("Summarization failed: %s", err)
        return _truncate_fallback(text)
