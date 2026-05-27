"""Gradio Emergency Command Center dashboard for incident triage."""

from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import quote

import gradio as gr
import pandas as pd

from app.ai_engines.pipeline import process_incident
from app.ai_engines.predictive_ml import get_classification_benchmark, list_districts
from app.core.config import settings
from app.repository.database import fetch_query_history

_HAZARD_CHOICES = [
    "",
    "flood",
    "drought",
    "landslide",
    "earthquake",
    "wildfire",
    "epidemic",
    "conflict",
    "other",
]

_RISK_BADGES = {
    "high": "🔴 HIGH RISK LEVEL DETECTED",
    "medium": "🟡 MEDIUM RISK LEVEL",
    "low": "🟢 LOW RISK LEVEL",
}

_MAP_PLACEHOLDER = (
    '<div class="gis-live-panel gis-live-panel--idle">'
    "<p class='gis-idle-msg'>Awaiting triage — operational map will render here.</p>"
    "</div>"
)

_COMMAND_CENTER_CSS = """
.command-header { letter-spacing: 0.04em; }
.gis-section { margin-top: 0.25rem; }
.gis-live-panel {
    border: 2px solid #38bdf8;
    border-radius: 10px;
    padding: 4px;
    background: #020617;
    min-height: 520px;
    box-shadow: 0 0 32px rgba(56, 189, 248, 0.12), inset 0 0 40px rgba(15, 23, 42, 0.95);
}
.gis-live-panel--idle { border-color: #475569; box-shadow: inset 0 0 24px rgba(15, 23, 42, 0.9); }
.gis-map-frame {
    width: 100%;
    height: 500px;
    border: none;
    display: block;
    background: #0f172a;
}
.gis-idle-msg { color: #94a3b8; padding: 3rem 1rem; text-align: center; margin: 0; }
.metrics-panel textarea { font-family: ui-monospace, monospace; font-size: 0.95rem; line-height: 1.45; }
"""


def _format_risk_metrics(risk: str, confidence: float) -> str:
    level = (risk or "").strip().lower()
    badge = _RISK_BADGES.get(level, f"⚪ {risk.upper()} RISK LEVEL")
    pct = confidence * 100.0 if confidence <= 1.0 else confidence
    pct = max(0.0, min(pct, 100.0))
    bar_len = 24
    filled = round(pct / 100.0 * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)

    train_acc, sla_target = get_classification_benchmark()
    sla_pct = sla_target * 100.0
    train_pct = train_acc * 100.0
    train_delta = train_pct - sla_pct
    train_meets = train_acc >= sla_target
    train_icon = "✓" if train_meets else "✗"

    # 3-class softmax: uniform guess ≈ 33%; scores in the 35–50% range are typical.
    chance_pct = 100.0 / len(settings.RISK_LABELS)
    conf_delta = pct - chance_pct
    conf_icon = "✓" if conf_delta >= 0 else "▼"

    return (
        f"{badge}\n\n"
        f"🎯 Prediction confidence: {pct:.1f}%\n"
        f"[{bar}] {pct:.1f}%\n"
        f"📏 vs. chance baseline ({chance_pct:.1f}%): "
        f"{conf_icon} {conf_delta:+.1f} pp\n\n"
        f"🧪 Model train accuracy: {train_pct:.1f}%\n"
        f"📐 SLA benchmark (train): {sla_pct:.0f}%\n"
        f"{train_icon} {'MEETS' if train_meets else 'BELOW'} SLA "
        f"({train_delta:+.1f} pp)"
    )


def _embed_map_iframe(map_path: str) -> str:
    """Serve Folium via iframe — full HTML documents break when inlined in gr.HTML."""
    name = quote(Path(map_path).name)
    version = int(time.time() * 1000)
    return (
        f'<div class="gis-live-panel">'
        f'<iframe class="gis-map-frame" '
        f'src="/risk-map/{name}?v={version}" '
        f'title="Operational risk map" '
        f'sandbox="allow-scripts allow-same-origin allow-popups" '
        f'loading="lazy"></iframe>'
        f"</div>"
    )


def _run_analysis(
    district: str, raw_report: str, hazard_type: str | None
) -> tuple[str, str, str, pd.DataFrame]:
    if not district or not district.strip():
        raise gr.Error("District is required.")
    if not raw_report or not raw_report.strip():
        raise gr.Error("Field report is required.")
    record = process_incident(
        district.strip(),
        raw_report.strip(),
        (hazard_type or "").strip() or None,
    )
    a = record.analysis
    return (
        a.summary,
        _format_risk_metrics(a.predicted_risk, a.confidence_score),
        _embed_map_iframe(a.map_path),
        _history_dataframe(),
    )


def _history_dataframe() -> pd.DataFrame:
    rows = fetch_query_history(limit=20)
    if not rows:
        return pd.DataFrame(
            columns=[
                "id",
                "timestamp",
                "district",
                "predicted_risk",
                "confidence_score",
                "summary",
            ]
        )
    df = pd.DataFrame(rows)
    return df[
        ["id", "timestamp", "district", "predicted_risk", "confidence_score", "summary"]
    ]


def _command_center_theme() -> gr.themes.Soft:
    return (
        gr.themes.Soft(
            primary_hue=gr.themes.colors.slate,
            secondary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        )
        .set(
            body_background_fill="*neutral_950",
            body_background_fill_dark="*neutral_950",
            block_background_fill="*neutral_900",
            block_background_fill_dark="*neutral_900",
            block_border_width="1px",
            block_label_text_weight="600",
            button_primary_background_fill="*primary_600",
            button_primary_background_fill_hover="*primary_500",
        )
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title=f"{settings.APP_TITLE} — Command Center",
        theme=_command_center_theme(),
        css=_COMMAND_CENTER_CSS,
    ) as demo:
        gr.Markdown(
            f"# 🚨 Emergency Command Center\n"
            f"### {settings.APP_TITLE}\n"
            "Unified triage console: **NLP summarization → risk classification → live geospatial tracking**, "
            "with every run appended to the system audit trail.",
            elem_classes=["command-header"],
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                gr.Markdown("## 📥 Ingest — Inbound Request Payload")
                with gr.Group():
                    district = gr.Dropdown(
                        choices=list_districts(),
                        label="Target District",
                        allow_custom_value=True,
                        info="Operational area for this incident report",
                    )
                    hazard_type = gr.Dropdown(
                        choices=_HAZARD_CHOICES,
                        label="Hazard Type",
                        value="",
                        info="Optional — classifies the inbound threat vector",
                    )
                    raw_report = gr.Textbox(
                        label="Raw Field Report",
                        lines=12,
                        placeholder="Paste observer notes, sensor excerpts, or dispatch transcripts…",
                    )
                    analyze_btn = gr.Button(
                        "🚀 Execute Triage Protocol",
                        variant="primary",
                        size="lg",
                    )

            with gr.Column(scale=1):
                gr.Markdown("## 📊 Telemetry — Real-Time AI Response")
                metrics_display = gr.Textbox(
                    interactive=False,
                    lines=12,
                    max_lines=14,
                    placeholder="Risk badge and model certainty appear here after triage…",
                    elem_classes=["metrics-panel"],
                    show_label=False,
                )
                with gr.Accordion("📝 Generated Summary (NLP)", open=True):
                    summary = gr.Textbox(
                        interactive=False,
                        lines=10,
                        max_lines=20,
                        placeholder="Condensed field report will appear here…",
                        show_label=False,
                    )

        with gr.Column(elem_classes=["gis-section"]):
            gr.Markdown("## 🗺️ Live GIS — Situational Awareness Display")
            with gr.Group():
                risk_map = gr.HTML(value=_MAP_PLACEHOLDER, show_label=False)

        gr.Markdown("## 📋 System Audit Trail Logging")
        history_df = gr.Dataframe(
            label="Recent query_log entries (newest runs appear after each triage)",
            interactive=False,
            wrap=True,
        )

        analyze_btn.click(
            _run_analysis,
            inputs=[district, raw_report, hazard_type],
            outputs=[summary, metrics_display, risk_map, history_df],
            show_progress="full",
        )
        demo.load(_history_dataframe, outputs=history_df)

    return demo
