"""Generative NLP engine for text summarization and incident dispatch synthesis."""

from __future__ import annotations

import logging
import re
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

def _output_lengths(input_tokens: int) -> tuple[int, int]:
    # Keep outputs well below input length to reduce extractive copy behavior.
    max_len = min(settings.SUMMARIZATION_MAX_LENGTH, max(32, input_tokens // 3))
    min_len = min(settings.SUMMARIZATION_MIN_LENGTH, max(16, max_len // 4))
    if min_len >= max_len:
        min_len = max(1, max_len - 1)
    return max_len, min_len


def _is_extractive_echo(summary: str, source: str) -> bool:
    a = summary.strip().lower()
    b = source.strip().lower()
    if len(a) < 24:
        return False
    probe = a[: min(len(a), 120)]
    return b.startswith(probe) or probe in b[: len(probe) + 80]


def _heuristic_summary(text: str, max_chars: int = 480) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if not sentences:
        return _truncate_fallback(text)
    parts: list[str] = []
    length = 0
    for sentence in sentences:
        if length + len(sentence) > max_chars and parts:
            break
        parts.append(sentence)
        length += len(sentence) + 1
    if parts:
        return " ".join(parts)
    return _truncate_fallback(text)


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

        def _run(max_l: int, min_l: int) -> str:
            result = pipe(
                model_input,
                max_length=max_l,
                min_length=min_l,
                do_sample=False,
            )
            return str(result[0]["summary_text"]).strip()

        summary = _run(max_len, min_len)
        if summary and _is_extractive_echo(summary, text):
            retry_max = max(24, max_len // 2)
            retry_min = min(min_len, max(8, retry_max // 3))
            if retry_min >= retry_max:
                retry_min = max(1, retry_max - 1)
            summary = _run(retry_max, retry_min)

        if summary and summary != model_input and not _is_extractive_echo(summary, text):
            return summary
        return _heuristic_summary(text)
    except (OSError, RuntimeError, ValueError, IndexError) as err:
        logger.warning(
            "Summarization failed (%s: %s); using truncated original text",
            type(err).__name__,
            err,
        )
        return _truncate_fallback(cleaned_report)
