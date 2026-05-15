"""
BERTopic Thematic Analysis Agent — Production Gradio UI
========================================================
A dashboard-style Gradio interface for orchestrating BERTopic topic modelling
via an LLM-backed agent defined in agent.py.

Layout
------
- Top: Header + Phase progress bar
- Body: Vertical cards in sequence
    1) Data Input
    2) Agent Console
    3) Results (Tabs: Review | Charts | Downloads)

Fixes applied (v2)
------------------
- BUG 3   : submit_review() now writes parsed review rows into
            agent_state["review_df"] BEFORE calling the agent, so
            _parse_review_df() in agent.py always receives a populated list.
- ISSUE 2 : PHASES list updated to 7 labels matching the actual B&C phases
            (was 6 labels misaligned with agent phase 0-6 mapping).
- ISSUE 4 : Added a startup API-key warning banner rendered in the UI when
            MISTRAL_API_KEY is not set in the environment.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import gradio as gr
import pandas as pd
import json
import os
import shutil
import uuid
from pathlib import Path
from urllib.parse import quote

# ---------------------------------------------------------------------------
# Agent import — graceful stub when agent.py is absent during dev/testing
# ---------------------------------------------------------------------------
try:
    from agent import agent
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False

    class _StubAgent:
        """Minimal stub so the UI works without agent.py."""

        def invoke(self, message: str, state: dict) -> tuple[str, dict]:
            reply = (
                f"[STUB] Received: **{message}**\n\n"
                "Connect `agent.py` to get real responses. "
                f"Current phase: `{state.get('phase', 0)}`."
            )
            state["phase"] = min(state.get("phase", 0) + 1, 8)
            return reply, state

    agent = _StubAgent()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# FIX ISSUE 2 — 7 labels aligned to the agent's phase 1-6 (index = phase-1)
PHASES = [
    "Familiarisation",   # Phase 1
    "Initial Codes",     # Phase 2
    "Themes",            # Phase 3
    "Review Themes",     # Phase 4
    "Naming",            # Phase 5
    "PAJAIS Mapping",    # Phase 5.5
    "Report",            # Phase 6
]

CHART_OPTIONS = ["Intertopic Map", "Top Words", "Hierarchy", "Heatmap"]

REVIEW_COLUMNS = [
    "#", "Topic Label", "Top Evidence", "Sentences", "Papers",
    "Approve", "Rename To", "Reasoning",
]

EMPTY_REVIEW_DF = pd.DataFrame(columns=REVIEW_COLUMNS)

# FIX ISSUE 4 — detect missing API keys at startup
MISTRAL_KEY_MISSING = not bool(os.environ.get("MISTRAL_API_KEY", ""))
GROQ_KEY_MISSING = not bool(os.environ.get("GROQ_API_KEY", ""))
UPLOADS_DIR = Path("uploads")
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"

# ---------------------------------------------------------------------------
# Custom CSS — SaaS dashboard aesthetic
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* Fonts */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=DM+Mono:wght@400;500&display=swap');

/* Tokens */
:root {
    --bg-base:          #0f1117;
    --bg-surface:       #181c27;
    --bg-elevated:      #1f2437;
    --bg-hover:         #252b3d;
    --border:           #2a3048;
    --border-active:    #4f6ef7;
    --text-primary:     #e8eaf0;
    --text-secondary:   #8b92a8;
    --text-muted:       #555f7a;
    --accent:           #4f6ef7;
    --accent-soft:      rgba(79,110,247,0.15);
    --accent-glow:      rgba(79,110,247,0.35);
    --success:          #34d399;
    --success-soft:     rgba(52,211,153,0.15);
    --warning:          #fbbf24;
    --warning-soft:     rgba(251,191,36,0.15);
    --danger:           #f87171;
    --radius-sm:        8px;
    --radius-md:        14px;
    --radius-lg:        20px;
    --shadow-card:      0 4px 24px rgba(0,0,0,0.45), 0 1px 3px rgba(0,0,0,0.3);
    --shadow-button:    0 2px 12px rgba(79,110,247,0.4);
    --font-ui:          'DM Sans', system-ui, sans-serif;
    --font-mono:        'DM Mono', 'Fira Code', monospace;
    --transition:       0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

body, .gradio-container {
    background: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-ui) !important;
}
.gradio-container { max-width: 1600px !important; padding: 0 !important; }

/* Header */
#app-header {
    background: linear-gradient(135deg, #0f1117 0%, #181c27 50%, #1a1f32 100%);
    border-bottom: 1px solid var(--border);
    padding: 24px 36px 20px;
    position: relative;
    overflow: hidden;
}
#app-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(79,110,247,0.18) 0%, transparent 70%);
    pointer-events: none;
}
#app-header .header-title {
    font-size: 1.7rem; font-weight: 700; letter-spacing: -0.03em;
    color: var(--text-primary); margin: 0 0 4px;
}
#app-header .header-subtitle {
    font-size: 0.875rem; color: var(--text-secondary); margin: 0;
}
#app-header .header-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--accent-soft); border: 1px solid var(--accent);
    border-radius: 100px; padding: 3px 12px; font-size: 0.75rem;
    font-weight: 600; color: var(--accent); margin-left: 12px; vertical-align: middle;
}

/* API key warning banner */
.api-warning {
    background: var(--warning-soft);
    border: 1px solid var(--warning);
    border-radius: var(--radius-sm);
    padding: 10px 16px;
    font-size: 0.83rem;
    font-weight: 500;
    color: var(--warning);
    margin: 12px 28px 0;
}

/* Phase progress bar */
.phase-bar-wrap {
    display: flex; align-items: center; gap: 0;
    margin-top: 20px; position: relative;
}
.phase-bar-wrap::before {
    content: '';
    position: absolute;
    left: 20px; right: 20px; top: 50%;
    height: 2px; background: var(--border);
    transform: translateY(-50%); z-index: 0;
}
.phase-item {
    display: flex; flex-direction: column;
    align-items: center; flex: 1; position: relative; z-index: 1;
}
.phase-dot {
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: 700;
    border: 2px solid var(--border); background: var(--bg-base);
    transition: all var(--transition);
}
.phase-dot.done   { background: var(--success-soft); border-color: var(--success); color: var(--success); }
.phase-dot.active { background: var(--accent-soft); border-color: var(--accent); color: var(--accent);
                    box-shadow: 0 0 14px var(--accent-glow); }
.phase-dot.pending { color: var(--text-muted); }
.phase-label {
    font-size: 0.65rem; font-weight: 500; color: var(--text-muted);
    margin-top: 6px; text-align: center; letter-spacing: 0.02em; white-space: nowrap;
}
.phase-label.active { color: var(--accent); }
.phase-label.done   { color: var(--success); }

/* Main body */
#main-body {
    padding: 22px 28px 32px;
    gap: 16px !important;
    max-width: 1160px;
    margin: 0 auto;
    width: 100%;
}

.panel-card {
    background:
        radial-gradient(1200px 260px at 100% -15%, rgba(79,110,247,0.12), transparent 52%),
        linear-gradient(180deg, rgba(31,36,55,0.9) 0%, rgba(24,28,39,0.95) 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-card);
    padding: 18px 18px 16px;
    position: relative;
    overflow: hidden;
    margin-bottom: 2px;
}

.panel-card:last-child { margin-bottom: 0; }

.panel-card::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(120deg, rgba(255,255,255,0.02), transparent 25%, transparent 75%, rgba(255,255,255,0.02));
    pointer-events: none;
}

.panel-data { margin-bottom: 2px; }
.panel-chat { margin-bottom: 2px; }

/* Card titles */
.card-title {
    font-size: 0.74rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--text-muted);
    margin: 0 0 16px; display: flex; align-items: center; gap: 10px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 12px;
}
.card-title::before {
    content: '';
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 10px var(--accent-glow);
}
.card-title span { font-size: 1.02rem; color: var(--text-primary); letter-spacing: 0.01em; }

/* Stats */
.stats-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 12px;
}
.stat-card {
    background: var(--bg-elevated); border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 12px 14px;
}
.stat-value { font-size: 1.4rem; font-weight: 700; color: var(--text-primary); line-height: 1; }
.stat-label { font-size: 0.72rem; color: var(--text-muted); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
.stat-card.accent .stat-value { color: var(--accent); }
.stat-card.success .stat-value { color: var(--success); }

/* Status pill */
.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 12px; border-radius: 100px; font-size: 0.78rem; font-weight: 600; margin-top: 12px;
}
.status-pill.idle    { background: rgba(139,146,168,0.12); color: var(--text-secondary); }
.status-pill.ready   { background: var(--success-soft); color: var(--success); }
.status-pill.working { background: var(--accent-soft); color: var(--accent); }
.status-pill .dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.status-pill.working .dot { animation: pulse-dot 1.2s ease-in-out infinite; }
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.7); }
}

/* Chatbot */
#chatbot-container .chatbot {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
}
.message.user {
    background: var(--accent-soft) !important;
    border: 1px solid rgba(79,110,247,0.2) !important;
    border-radius: 14px 14px 4px 14px !important;
    color: var(--text-primary) !important;
    font-size: 0.875rem !important;
}
.message.bot {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px 14px 14px 4px !important;
    color: var(--text-primary) !important;
    font-size: 0.875rem !important;
}

/* Chat input */
#chat-input-row { display: flex; gap: 10px; margin-top: 12px; align-items: flex-end; }
#chat-input-row textarea {
    background: var(--bg-elevated) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important; color: var(--text-primary) !important;
    font-family: var(--font-ui) !important; font-size: 0.875rem !important;
    resize: none !important; transition: border-color var(--transition) !important;
}
#chat-input-row textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-soft) !important;
}

/* Buttons */
.btn-primary {
    background: var(--accent) !important; border: none !important;
    border-radius: var(--radius-sm) !important; color: #fff !important;
    font-family: var(--font-ui) !important; font-weight: 600 !important;
    font-size: 0.875rem !important; padding: 10px 20px !important;
    cursor: pointer !important; box-shadow: var(--shadow-button) !important;
    transition: all var(--transition) !important; white-space: nowrap;
}
.btn-primary:hover {
    background: #3d5de6 !important;
    box-shadow: 0 4px 20px rgba(79,110,247,0.55) !important;
    transform: translateY(-1px) !important;
}
.btn-primary:disabled { opacity: 0.45 !important; cursor: not-allowed !important; transform: none !important; }

.btn-secondary {
    background: var(--bg-elevated) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important; color: var(--text-secondary) !important;
    font-family: var(--font-ui) !important; font-weight: 500 !important;
    font-size: 0.875rem !important; padding: 10px 18px !important;
    cursor: pointer !important; transition: all var(--transition) !important;
}
.btn-secondary:hover {
    background: var(--bg-hover) !important; border-color: var(--accent) !important;
    color: var(--text-primary) !important;
}

.btn-success {
    background: rgba(52,211,153,0.15) !important; border: 1px solid var(--success) !important;
    border-radius: var(--radius-sm) !important; color: var(--success) !important;
    font-family: var(--font-ui) !important; font-weight: 600 !important;
    font-size: 0.875rem !important; padding: 10px 20px !important;
    cursor: pointer !important; transition: all var(--transition) !important;
}
.btn-success:hover { background: rgba(52,211,153,0.25) !important; box-shadow: 0 2px 14px rgba(52,211,153,0.3) !important; }

/* Tabs */
.tabs > .tab-nav {
    background: var(--bg-elevated) !important; border-bottom: 1px solid var(--border) !important;
    border-radius: var(--radius-md) var(--radius-md) 0 0 !important;
    padding: 6px 6px 0 !important; gap: 4px !important;
}
.tabs > .tab-nav button {
    background: transparent !important; border: none !important;
    color: var(--text-muted) !important; font-family: var(--font-ui) !important;
    font-size: 0.8rem !important; font-weight: 600 !important;
    letter-spacing: 0.04em !important; padding: 8px 16px !important;
    border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
    transition: all var(--transition) !important; cursor: pointer !important;
}
.tabs > .tab-nav button:hover { color: var(--text-primary) !important; background: var(--bg-hover) !important; }
.tabs > .tab-nav button.selected {
    color: var(--accent) !important; background: var(--accent-soft) !important;
    box-shadow: inset 0 -2px 0 var(--accent) !important;
}
.tabitem {
    background: var(--bg-elevated) !important; border: 1px solid var(--border) !important;
    border-top: none !important; border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
    padding: 16px !important;
}

/* Dataframe */
.dataframe-wrap {
    overflow-x: auto !important;
}
.dataframe-wrap table {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    border-collapse: collapse !important;
    width: max-content !important;
    min-width: 100% !important;
    table-layout: auto !important;
}
.dataframe-wrap th {
    background: var(--bg-elevated) !important; color: var(--text-muted) !important;
    font-family: var(--font-ui) !important; font-size: 0.72rem !important;
    font-weight: 600 !important; letter-spacing: 0.06em !important;
    text-transform: uppercase !important; padding: 10px 12px !important;
    border-bottom: 1px solid var(--border) !important;
    min-width: 120px !important;
}
.dataframe-wrap td {
    background: var(--bg-surface) !important; color: var(--text-primary) !important;
    padding: 9px 12px !important; border-bottom: 1px solid var(--border) !important;
    line-height: 1.35 !important;
    vertical-align: top !important;
    min-width: 120px !important;
}
.dataframe-wrap th,
.dataframe-wrap td {
    white-space: nowrap !important;
}
.dataframe-wrap td > div,
.dataframe-wrap td > span,
.dataframe-wrap td > p {
    display: block !important;
    max-width: none !important;
    white-space: nowrap !important;
    overflow: visible !important;
    text-overflow: clip !important;
    cursor: pointer !important;
}
.dataframe-wrap td:focus-within > div,
.dataframe-wrap td:focus-within > span,
.dataframe-wrap td:focus-within > p {
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
.dataframe-wrap textarea,
.dataframe-wrap input[type="text"] {
    white-space: nowrap !important;
    overflow-wrap: normal !important;
    word-break: normal !important;
    overflow-x: auto !important;
    width: 100% !important;
    min-width: 160px !important;
    box-sizing: border-box !important;
}
.dataframe-wrap textarea {
    min-height: 38px !important;
    height: 38px !important;
    max-height: 38px !important;
    overflow-y: hidden !important;
    resize: none !important;
}
.dataframe-wrap tr:hover td { background: var(--bg-hover) !important; }
.dataframe-wrap input[type="checkbox"] {
    appearance: auto !important;
    accent-color: var(--accent) !important;
    cursor: pointer !important;
    width: 16px;
    height: 16px;
}

/* Chart frame */
.chart-frame {
    width: 100%; min-height: 420px; border: 1px solid var(--border);
    border-radius: var(--radius-md); background: var(--bg-elevated); overflow: hidden;
}

/* Vertical card spacing on small screens */
@media (max-width: 900px) {
    #main-body {
        padding: 14px 12px 20px;
        gap: 12px !important;
    }
    .panel-card {
        padding: 14px 12px;
        border-radius: var(--radius-md);
    }
    .chart-frame { min-height: 320px; }
}

/* Download list */
.file-list-item {
    display: flex; align-items: center; gap: 10px;
    background: var(--bg-elevated); border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 10px 14px; margin-bottom: 8px;
    transition: all var(--transition);
}
.file-list-item:hover { border-color: var(--accent); background: var(--bg-hover); }
.file-icon { font-size: 1.1rem; }
.file-name { font-size: 0.83rem; color: var(--text-primary); flex: 1; font-family: var(--font-mono); }
.file-size { font-size: 0.72rem; color: var(--text-muted); }

/* Misc Gradio overrides */
label, .label-wrap { color: var(--text-secondary) !important; font-family: var(--font-ui) !important; font-size: 0.8rem !important; }
input:not([type="checkbox"]), textarea { background: var(--bg-elevated) !important; color: var(--text-primary) !important; border-color: var(--border) !important; }
.gr-form:not(.panel-card), .gr-box:not(.panel-card) { background: transparent !important; border: none !important; }
footer { display: none !important; }
select { background: var(--bg-elevated) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important; color: var(--text-primary) !important; font-family: var(--font-ui) !important; font-size: 0.875rem !important; padding: 8px 12px !important; }

/* Animations */
.fade-in { animation: fadeIn 0.35s ease-out both; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }

/* Scrollbar */
::-webkit-scrollbar       { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: #2d3550; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3d4770; }
"""

# ---------------------------------------------------------------------------
# Helper — build phase-progress HTML
# FIX ISSUE 2 — phase index maps correctly to 7-item PHASES list
# ---------------------------------------------------------------------------
def build_phase_html(current_phase: int) -> str:
    """
    Render the 7-step phase progress bar.
    current_phase is the agent's phase (1-7); phase 0 = no phase started yet.
    Phase 8 indicates full completion and renders all 7 steps as done.
    """
    items = []
    for i, label in enumerate(PHASES):
        phase_number = i + 1    # phases are 1-indexed
        if phase_number < current_phase:
            dot_cls, lbl_cls, icon = "done",   "done",   "v"
        elif phase_number == current_phase:
            dot_cls, lbl_cls, icon = "active", "active", str(phase_number)
        else:
            dot_cls, lbl_cls, icon = "pending", "",       str(phase_number)

        items.append(f"""
        <div class="phase-item">
            <div class="phase-dot {dot_cls}">{icon}</div>
            <div class="phase-label {lbl_cls}">{label}</div>
        </div>""")

    inner = "\n".join(items)
    return f"""
    <div id="app-header">
        <div style="display:flex;align-items:baseline;gap:4px;">
            <span class="header-title">BERTopic Thematic Analysis Agent</span>
            <span class="header-badge">AI-Powered</span>
        </div>
        <p class="header-subtitle">
            End-to-end topic modelling — upload a Scopus corpus, run the agent, review topics.
        </p>
        <div class="phase-bar-wrap">
            {inner}
        </div>
    </div>"""


# ---------------------------------------------------------------------------
# Helper — dataset stats HTML
# ---------------------------------------------------------------------------
def build_stats_html(rows: int, cols: int, filename: str) -> str:
    return f"""
    <div class="stats-grid fade-in">
        <div class="stat-card accent">
            <div class="stat-value">{rows:,}</div>
            <div class="stat-label">Rows</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{cols}</div>
            <div class="stat-label">Columns</div>
        </div>
    </div>
    <div class="status-pill ready" style="margin-top:14px;">
        <div class="dot"></div>
        {filename}
    </div>"""


# ---------------------------------------------------------------------------
# Helper — download file-list HTML
# ---------------------------------------------------------------------------
def build_file_list_html(paths: list[str]) -> str:
    if not paths:
        return "<p style='color:var(--text-muted);font-size:0.83rem;padding:8px 0;'>No files generated yet.</p>"
    icons = {".csv": "CSV", ".json": "JSON", ".html": "HTML", ".png": "IMG", ".xlsx": "XLS", ".txt": "TXT"}
    items = []
    for p in paths:
        p    = Path(p)
        ext  = p.suffix.lower()
        icon = icons.get(ext, "FILE")
        size = ""
        if p.exists():
            b    = p.stat().st_size
            size = f"{b/1024:.1f} KB" if b < 1_048_576 else f"{b/1_048_576:.1f} MB"
        items.append(f"""
        <div class="file-list-item fade-in">
            <span class="file-icon" style="font-size:0.7rem;background:var(--accent-soft);color:var(--accent);
                  padding:2px 5px;border-radius:4px;font-family:var(--font-mono);font-weight:600;">{icon}</span>
            <span class="file-name">{p.name}</span>
            <span class="file-size">{size}</span>
        </div>""")
    return "\n".join(items)


# ---------------------------------------------------------------------------
# Helper — placeholder chart HTML
# ---------------------------------------------------------------------------
def build_placeholder_chart(chart_type: str) -> str:
    colour_map = {
        "Intertopic Map": "#4f6ef7",
        "Top Words":      "#34d399",
        "Hierarchy":      "#fbbf24",
        "Heatmap":        "#f87171",
    }
    col = colour_map.get(chart_type, "#4f6ef7")
    return f"""
    <div class="chart-frame" style="display:flex;align-items:center;justify-content:center;flex-direction:column;gap:10px;">
        <div style="font-size:2rem;color:var(--text-muted);">CHART</div>
        <div style="color:var(--text-secondary);font-size:0.9rem;font-weight:600;">{chart_type}</div>
        <div style="color:var(--text-muted);font-size:0.78rem;">Run the agent to generate this chart.</div>
        <div style="width:180px;height:4px;background:var(--border);border-radius:2px;margin-top:6px;">
            <div style="width:0%;height:4px;background:{col};border-radius:2px;animation:grow 2s ease-in-out infinite alternate;"></div>
        </div>
    </div>
    <style>@keyframes grow {{ from{{width:0%}} to{{width:75%}} }}</style>"""


# ---------------------------------------------------------------------------
# Core interaction handlers
# ---------------------------------------------------------------------------

def _persist_upload(file_obj) -> Path:
    """Copy Gradio temp upload to a stable local path and return it."""
    src = Path(file_obj.name)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    dst = UPLOADS_DIR / f"{uuid.uuid4().hex[:10]}_{src.name}"
    shutil.copy2(src, dst)
    return dst.resolve()

def handle_file_upload(file_obj, agent_state):
    """Parse uploaded CSV, store file_path in state, trigger agent."""
    if file_obj is None:
        return (
            "<p style='color:var(--text-muted);font-size:0.83rem;'>No file selected.</p>",
            "<div class='status-pill idle'><div class='dot'></div>Awaiting upload</div>",
            agent_state,
            build_phase_html(agent_state.get("phase", 0)),
        )

    try:
        persisted = _persist_upload(file_obj)
        df       = pd.read_csv(persisted)
        rows, cols = df.shape
        filename = Path(file_obj.name).name
        stats_html = build_stats_html(rows, cols, filename)
        agent_state["file_path"] = str(persisted)
        agent_state["file_name"] = filename
        agent_state["rows"]      = rows
        agent_state["cols"]      = cols
    except Exception as exc:
        stats_html = f"<p style='color:var(--danger);font-size:0.83rem;'>Upload error: {exc}</p>"

    status_html = "<div class='status-pill ready'><div class='dot'></div>File ready</div>"
    phase_html  = build_phase_html(agent_state.get("phase", 0))
    return stats_html, status_html, agent_state, phase_html


def handle_chat(user_message: str, chat_history: list, agent_state: dict):
    """Stream one user turn through the agent."""
    if not user_message.strip():
        yield chat_history, agent_state, build_phase_html(agent_state.get("phase", 0))
        return

    chat_history = chat_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": "Thinking..."},
    ]
    yield chat_history, agent_state, build_phase_html(agent_state.get("phase", 0))

    file_path = agent_state.get("file_path")
    if file_path and not Path(file_path).exists():
        chat_history[-1]["content"] = (
            "Uploaded CSV is no longer available on disk. "
            "Please upload the file again and retry."
        )
        yield chat_history, agent_state, build_phase_html(agent_state.get("phase", 0))
        return

    try:
        reply, agent_state = agent.invoke(user_message, agent_state)
    except Exception as exc:
        reply = f"Agent error: `{exc}`"

    chat_history[-1]["content"] = reply
    yield chat_history, agent_state, build_phase_html(agent_state.get("phase", 0))


def auto_trigger_agent(agent_state: dict, chat_history: list):
    """Fire an automatic Phase 1 trigger after file upload."""
    filename = agent_state.get("file_name", "uploaded file")
    rows     = agent_state.get("rows", 0)
    auto_msg = (
        f"A dataset has been uploaded: **{filename}** ({rows:,} rows). "
        "Please start the thematic analysis pipeline."
    )
    results = []
    for state in handle_chat(auto_msg, chat_history, agent_state):
        results = state
    return results   # (chat_history, agent_state, phase_html)


def refresh_review_table(agent_state: dict):
    """Render the review DataFrame from agent_state."""
    raw = agent_state.get("review_df", [])
    if raw:
        try:
            return gr.update(value=pd.DataFrame(raw), interactive=True)
        except Exception:
            pass
    return gr.update(value=EMPTY_REVIEW_DF.copy(), interactive=True)


def submit_review(review_df, agent_state: dict, chat_history: list):
    """
    FIX BUG 3 — write parsed review rows into agent_state["review_df"]
    BEFORE calling the agent, so _parse_review_df() receives the populated list.
    """
    def _next_phase_message(state: dict) -> str:
        gate = state.get("stop_gate")
        if gate == "STOP_GATE_1_AWAIT_REVIEW_TABLE":
            return "Review table submitted. Please proceed to Phase 3 and consolidate themes."
        if gate == "STOP_GATE_2_AWAIT_THEME_MERGE":
            return "Theme merge confirmed. Please proceed to Phase 4 for saturation check."
        if gate == "STOP_GATE_3_AWAIT_SATURATION_SIGNOFF":
            return "Saturation sign-off confirmed. Please proceed to Phase 5 for naming themes."
        if gate == "STOP_GATE_4_AWAIT_TAXONOMY_REVIEW":
            return "Taxonomy review confirmed. Please proceed to Phase 6 to finalize outputs."
        return "Review table submitted. Please proceed to the next phase."

    # Store the review table in state so agent.py can read it
    agent_state["review_df"] = review_df.to_dict(orient="records")
    agent_state["review_submitted"] = True

    # Send a short trigger message — the agent reads state, not the payload
    msg = _next_phase_message(agent_state)
    results = []
    for state in handle_chat(msg, chat_history, agent_state):
        results = state
    new_history, new_state, phase_html = results
    return new_history, new_state, phase_html


def auto_accept_review(agent_state: dict, chat_history: list, enabled: bool):
    """Auto-approve Phase 2 review rows and submit when enabled."""
    if not enabled:
        return chat_history, agent_state, build_phase_html(agent_state.get("phase", 0))

    gate = agent_state.get("stop_gate")
    if gate != "STOP_GATE_1_AWAIT_REVIEW_TABLE":
        return chat_history, agent_state, build_phase_html(agent_state.get("phase", 0))

    if agent_state.get("review_submitted"):
        return chat_history, agent_state, build_phase_html(agent_state.get("phase", 0))

    if agent_state.get("auto_accept_last_gate") == gate:
        return chat_history, agent_state, build_phase_html(agent_state.get("phase", 0))

    rows = agent_state.get("review_df", [])
    if not rows:
        return chat_history, agent_state, build_phase_html(agent_state.get("phase", 0))

    df = pd.DataFrame(rows)
    if "Approve" in df.columns:
        df["Approve"] = True
    if "Rename To" in df.columns and "Topic Label" in df.columns:
        df["Rename To"] = df["Rename To"].fillna("").astype(str)
        df["Rename To"] = df.apply(
            lambda r: r["Rename To"] or r["Topic Label"], axis=1
        )

    new_history, new_state, phase_html = submit_review(df, agent_state, chat_history)
    new_state["auto_accept_last_gate"] = gate
    return new_history, new_state, phase_html


def refresh_downloads(agent_state: dict):
    """Return downloadable artefact paths from agent state."""
    files = agent_state.get("output_files", [])
    html  = build_file_list_html(files)
    valid = [f for f in files if os.path.exists(f)]
    return html, valid if valid else None


def get_chart_html(chart_choice: str, agent_state: dict) -> str:
    """Return chart iframe or placeholder HTML."""
    charts = agent_state.get("charts", {})
    if chart_choice in charts:
        src = charts[chart_choice]
        if os.path.exists(src):
            # Gradio 6 serves local files from /gradio_api/file=..., and
            # paths must be URL-encoded when directories contain spaces.
            normalised = str(Path(src).resolve()).replace("\\", "/")
            encoded = quote(normalised, safe="/:")
            return (
                f'<iframe src="./gradio_api/file={encoded}" '
                'class="chart-frame" frameborder="0"></iframe>'
            )
        return f'<div class="chart-frame fade-in">{src}</div>'
    return build_placeholder_chart(chart_choice)


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="BERTopic Thematic Analysis Agent",
    ) as app:

        # ── Shared state ──────────────────────────────────────────────────
        agent_state  = gr.State({})
        chat_history = gr.State([])

        # ── Header ───────────────────────────────────────────────────────
        phase_bar = gr.HTML(value=build_phase_html(0), elem_id="phase-bar")

        # FIX ISSUE 4 — show warning banner when API key is missing
        if MISTRAL_KEY_MISSING:
            gr.HTML(
                "<div class='api-warning'>"
                "WARNING: MISTRAL_API_KEY is not set. "
                "All LLM calls will fail. "
                "Set it in HuggingFace Spaces: Settings -> Variables and secrets."
                "</div>"
            )

        if GROQ_KEY_MISSING:
            gr.HTML(
                "<div class='api-warning'>"
                "WARNING: GROQ_API_KEY is not set. "
                "VERIFY command will be unavailable for Groq side-by-side checks. "
                "Set it to enable Mistral + Groq-Ollama + Groq-GPT verification in Phase 2 "
                "and Groq verification in Phase 5.5."
                "</div>"
            )

        # ── Main vertical body ────────────────────────────────────────────
        with gr.Column(elem_id="main-body"):

            with gr.Column(elem_classes=["panel-card", "panel-data"]):
                gr.HTML("""<div class="card-title"><span>Data Input</span></div>""")

                file_input = gr.File(
                    label="Upload Corpus (CSV)",
                    file_types=[".csv"],
                    interactive=True,
                    elem_id="csv-upload",
                )

                file_status = gr.HTML(
                    value="<div class='status-pill idle'><div class='dot'></div>Awaiting upload</div>"
                )

                dataset_stats = gr.HTML(
                    value="<p style='color:var(--text-muted);font-size:0.83rem;"
                          "padding:8px 0 0;'>Upload a CSV to see statistics.</p>"
                )

                gr.HTML("<hr style='border:none;border-top:1px solid var(--border);margin:16px 0;'>")
                gr.HTML("""
                <div style='font-size:0.72rem;color:var(--text-muted);line-height:1.7;'>
                    <b style='color:var(--text-secondary);'>Expected columns</b><br>
                    Title, Abstract, Author Keywords, Authors, Year<br><br>
                    <b style='color:var(--text-secondary);'>Quick commands</b><br>
                    <code style='font-family:var(--font-mono);'>run abstract</code><br>
                    <code style='font-family:var(--font-mono);'>run title</code><br>
                    <code style='font-family:var(--font-mono);'>run keywords</code><br>
                    <code style='font-family:var(--font-mono);'>verify</code><br>
                    <code style='font-family:var(--font-mono);'>show topics</code><br>
                    <code style='font-family:var(--font-mono);'>export results</code>
                </div>""")

            with gr.Column(elem_classes=["panel-card", "panel-chat"]):
                gr.HTML("""<div class="card-title"><span>Agent Console</span></div>""")

                chatbot = gr.Chatbot(
                    value=[],
                    height=470,
                    show_label=False,
                    avatar_images=(None, None),
                    elem_id="chatbot-container",
                )

                with gr.Row(elem_id="chat-input-row"):
                    chat_input = gr.Textbox(
                        placeholder='Type a command, e.g. "run abstract" or "run keywords" ...',
                        show_label=False,
                        lines=1,
                        scale=5,
                        container=False,
                    )
                    send_btn = gr.Button(
                        "Send",
                        variant="primary",
                        scale=1,
                        min_width=90,
                        elem_classes=["btn-primary"],
                    )

                with gr.Row():
                    clear_btn = gr.Button(
                        "Clear Chat",
                        variant="secondary",
                        scale=1,
                        elem_classes=["btn-secondary"],
                    )

            with gr.Column(elem_classes=["panel-card", "panel-results"]):
                gr.HTML("""<div class="card-title"><span>Results</span></div>""")

                with gr.Tabs(elem_classes=["tabs"]):

                    # ── Tab 1: Review Table ─────────────────────────────
                    with gr.TabItem("Review", elem_classes=["tabitem"]):
                        gr.HTML("""
                        <p style='font-size:0.78rem;color:var(--text-muted);margin:0 0 12px;'>
                            Edit <b>Approve</b>, <b>Rename To</b>, and <b>Reasoning</b> columns inline,
                            and use the <b>Papers</b> column to see the top 3 paper titles per cluster.
                            then click <b>Submit Review</b>. Use <b>verify</b> in chat at Phase 2
                            or Phase 5.5 to see Mistral vs Groq comparisons directly in chat output.
                            Phase 2 verification also adds an adjudicated best label.
                            Enable <b>Auto-accept Phase 2 review</b> to skip manual submission.
                        </p>""")

                        review_table = gr.Dataframe(
                            value=EMPTY_REVIEW_DF.copy(),
                            headers=REVIEW_COLUMNS,
                            datatype=[
                                "number", "str", "str", "number", "str",
                                "bool", "str", "str",
                            ],
                            interactive=True,
                            wrap=False,
                            elem_classes=["dataframe-wrap"],
                        )

                        with gr.Row():
                            refresh_table_btn = gr.Button(
                                "Refresh",
                                variant="secondary",
                                scale=1,
                                elem_classes=["btn-secondary"],
                            )
                            submit_review_btn = gr.Button(
                                "Submit Review",
                                variant="primary",
                                scale=2,
                                elem_classes=["btn-success"],
                            )

                        auto_accept_toggle = gr.Checkbox(
                            label="Auto-accept Phase 2 review and continue",
                            value=False,
                        )

                    # ── Tab 2: Charts ───────────────────────────────────
                    with gr.TabItem("Charts", elem_classes=["tabitem"]):
                        chart_selector = gr.Dropdown(
                            choices=CHART_OPTIONS,
                            value=CHART_OPTIONS[0],
                            label="Select chart",
                            interactive=True,
                        )
                        chart_display = gr.HTML(
                            value=build_placeholder_chart(CHART_OPTIONS[0])
                        )

                    # ── Tab 3: Downloads ────────────────────────────────
                    with gr.TabItem("Downloads", elem_classes=["tabitem"]):
                        gr.HTML("""
                        <p style='font-size:0.78rem;color:var(--text-muted);margin:0 0 12px;'>
                            Files generated by the agent will appear here automatically.
                        </p>""")

                        download_file_list_html = gr.HTML(
                            value="<p style='color:var(--text-muted);font-size:0.83rem;'>"
                                  "No files generated yet.</p>"
                        )

                        download_files = gr.File(
                            label="",
                            file_count="multiple",
                            interactive=False,
                        )

                        refresh_dl_btn = gr.Button(
                            "Refresh Downloads",
                            variant="secondary",
                            elem_classes=["btn-secondary"],
                        )

        # ────────────────────────────────────────────────────────────────
        # Event wiring
        # ────────────────────────────────────────────────────────────────

        def _on_file_upload(file_obj, a_state, c_history):
            stats, status, a_state, phase_html = handle_file_upload(file_obj, a_state)
            if file_obj is not None and "file_path" in a_state:
                c_history, a_state, phase_html = auto_trigger_agent(a_state, c_history)
            return stats, status, a_state, phase_html, c_history

        file_input.change(
            fn=_on_file_upload,
            inputs=[file_input, agent_state, chat_history],
            outputs=[dataset_stats, file_status, agent_state, phase_bar, chatbot],
        )

        def _on_send(msg, c_history, a_state):
            accumulated = []
            for result in handle_chat(msg, c_history, a_state):
                accumulated = result
                yield accumulated[0], accumulated[1], accumulated[2], ""

        send_btn.click(
            fn=_on_send,
            inputs=[chat_input, chatbot, agent_state],
            outputs=[chatbot, agent_state, phase_bar, chat_input],
        )
        chat_input.submit(
            fn=_on_send,
            inputs=[chat_input, chatbot, agent_state],
            outputs=[chatbot, agent_state, phase_bar, chat_input],
        )

        clear_btn.click(
            fn=lambda: ([], {}),
            outputs=[chatbot, agent_state],
        )

        refresh_table_btn.click(
            fn=refresh_review_table,
            inputs=[agent_state],
            outputs=[review_table],
        )

        # FIX BUG 3 — submit_review now writes review_df into state first
        submit_review_btn.click(
            fn=submit_review,
            inputs=[review_table, agent_state, chatbot],
            outputs=[chatbot, agent_state, phase_bar],
        )

        chart_selector.change(
            fn=get_chart_html,
            inputs=[chart_selector, agent_state],
            outputs=[chart_display],
        )

        refresh_dl_btn.click(
            fn=refresh_downloads,
            inputs=[agent_state],
            outputs=[download_file_list_html, download_files],
        )

        # Auto-refresh review table, downloads, and the active chart after every chat turn.
        chatbot.change(
            fn=lambda selected_chart, a: (
                refresh_review_table(a),
                *refresh_downloads(a),
                get_chart_html(selected_chart, a),
            ),
            inputs=[chart_selector, agent_state],
            outputs=[review_table, download_file_list_html, download_files, chart_display],
        )

        # Auto-accept Phase 2 review when enabled.
        chatbot.change(
            fn=auto_accept_review,
            inputs=[agent_state, chatbot, auto_accept_toggle],
            outputs=[chatbot, agent_state, phase_bar],
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        allowed_paths=[str(OUTPUTS_DIR.resolve())],
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.indigo,
            secondary_hue=gr.themes.colors.slate,
            neutral_hue=gr.themes.colors.slate,
            font=[gr.themes.GoogleFont("DM Sans"), "system-ui", "sans-serif"],
        ),
    )
