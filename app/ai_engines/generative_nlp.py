"""Generative NLP: field reports → concise summaries (Hugging Face BART)."""

from __future__ import annotations

import logging
import re
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
        logger.warning("Summarization failed: %s", err)
        return _truncate_fallback(text)
