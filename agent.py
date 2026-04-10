"""
agent.py — LangGraph BERTopic Thematic Analysis Agent
======================================================
A strictly phase-gated ReAct agent orchestrating Braun & Clarke's (2006)
six-phase thematic analysis pipeline via LangGraph.

Architecture
------------
- LLM        : ChatMistralAI (mistral-small-latest, free tier)
- Agent type : create_react_agent (LangGraph)
- Memory     : MemorySaver (in-process checkpointing)
- Tools      : 7 tools imported from tools.py
- State      : agent_state dict flows through app.py <-> agent.invoke()

Phase gating
------------
  Phase 0 -> awaiting file upload
  Phase 1 -> Familiarisation        [load_scopus_csv]
  Phase 2 -> Initial Codes          [run_bertopic_discovery, label_topics_with_llm]
             STOP GATE 1 — await review table submission
  Phase 3 -> Searching Themes       [consolidate_into_themes]
             STOP GATE 2 — await theme-merge confirmation
  Phase 4 -> Reviewing Themes       [saturation check via LLM]
             STOP GATE 3 — await researcher sign-off
  Phase 5 -> Defining & Naming      [final naming confirmation]
  Phase 5.5-> PAJAIS Mapping        [compare_with_taxonomy]
             STOP GATE 4 — await taxonomy review
  Phase 6 -> Report                 [generate_comparison_csv, export_narrative]

Fixes applied (v2)
------------------
- BUG 2   : Removed dead lambda block (lines 514-520 in v1) that ran
            _preprocess_phase3() twice, wasting an LLM call on every Phase 3
            trigger. The correct ternary expression is now the only path.
- ISSUE 3 : After Phase 2 labels are generated, _populate_review_df() converts
            labels.json into properly formatted review table rows and stores
            them in agent_state["review_df"] so app.py can render the table.
- ISSUE 4 : Added startup warning when MISTRAL_API_KEY is missing.

Integration contract (app.py)
------------------------------
  from agent import agent

  reply, new_state = agent.invoke(user_message, agent_state)

  agent_state keys consumed / produced:
    phase           int        current phase index (0-6)
    file_path       str        path to uploaded CSV
    run_key         str        "abstract" | "title"
    review_df       list[dict] review table rows (populated after Phase 2)
    theme_map       dict       {theme_name: [cluster_id, ...]}
    charts          dict       {chart_name: html_path}
    output_files    list[str]  paths to downloadable artefacts
    thread_id       str        LangGraph memory thread identifier
    stop_gate       str|None   active gate name or None
"""

# ---------------------------------------------------------------------------
# Stdlib
# ---------------------------------------------------------------------------
import os
import json
import uuid

# ---------------------------------------------------------------------------
# LangChain / LangGraph
# ---------------------------------------------------------------------------
from langchain_core.messages import HumanMessage
from langchain_mistralai import ChatMistralAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# ---------------------------------------------------------------------------
# Project tools
# ---------------------------------------------------------------------------
from tools import (
    ALL_TOOLS,
    OUTPUT_DIR,
    _load_json,
    _run_dir,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MISTRAL_API_KEY: str = os.environ.get("MISTRAL_API_KEY", "")
MODEL_NAME:      str = "mistral-small-latest"
DEFAULT_RUN_KEY: str = "abstract"
THREAD_PREFIX:   str = "TA-"

# FIX ISSUE 4 — surface missing API key immediately at import time
_KEY_MISSING = not bool(MISTRAL_API_KEY)
_KEY_MISSING and print(
    "\n[WARNING] MISTRAL_API_KEY is not set. "
    "All LLM calls will fail with HTTP 401.\n"
    "Set it via: export MISTRAL_API_KEY='your-key'\n"
    "On HuggingFace Spaces: Settings -> Variables and secrets\n"
)

# ---------------------------------------------------------------------------
# Stop gate identifiers
# ---------------------------------------------------------------------------
GATE_POST_PHASE2  = "STOP_GATE_1_AWAIT_REVIEW_TABLE"
GATE_POST_PHASE3  = "STOP_GATE_2_AWAIT_THEME_MERGE"
GATE_POST_PHASE4  = "STOP_GATE_3_AWAIT_SATURATION_SIGNOFF"
GATE_POST_PHASE55 = "STOP_GATE_4_AWAIT_TAXONOMY_REVIEW"

# ---------------------------------------------------------------------------
# Phase labels (used in progress reporting to app.py)
# ---------------------------------------------------------------------------
PHASE_LABELS = {
    0: "Awaiting Upload",
    1: "Phase 1 — Familiarisation",
    2: "Phase 2 — Initial Codes",
    3: "Phase 3 — Searching Themes",
    4: "Phase 4 — Reviewing Themes",
    5: "Phase 5 — Defining & Naming",
    6: "Phase 6 — Report",
}

# ============================================================================
# System prompt
# ============================================================================

SYSTEM_PROMPT = """
## ROLE
You are a **computational thematic analysis expert** specialising in BERTopic and
Braun & Clarke's (2006) six-phase reflexive thematic analysis methodology.
You orchestrate a structured research pipeline for academic literature review.

---

## CRITICAL BEHAVIOURAL RULES
1. **One phase per response** — complete exactly one phase, then STOP.
2. **NEVER skip phases** — every phase must execute in order: 1 -> 2 -> 3 -> 4 -> 5 -> 5.5 -> 6.
3. **NEVER auto-advance** — you must wait for an explicit user command before moving to the next phase.
4. **ALL topic approvals come from the review table** — NEVER approve, merge, or rename topics based on chat messages alone.
5. **NEVER hallucinate tool outputs** — only report what a tool actually returned.
6. **NEVER invent topic labels, themes, or statistics** — use only tool-generated data.
7. **Always end your response with the correct STOP instruction** (see phase definitions).

---

## AVAILABLE TOOLS

| # | Tool | Purpose | When to Use |
|---|------|---------|-------------|
| 1 | `load_scopus_csv` | Load & parse Scopus CSV; extract sentences; compute stats | Phase 1 only |
| 2 | `run_bertopic_discovery` | Embed sentences, cluster, extract evidence, generate 4 charts | Phase 2 only |
| 3 | `label_topics_with_llm` | Send cluster evidence to Mistral; get label/category/reasoning per cluster | Phase 2 only, after discovery |
| 4 | `consolidate_into_themes` | Merge approved clusters into named themes; recompute centroids | Phase 3 only, after review table submitted |
| 5 | `compare_with_taxonomy` | Map themes to PAJAIS 25-category taxonomy; classify MAPPED vs NOVEL | Phase 5.5 only |
| 6 | `generate_comparison_csv` | Side-by-side abstract vs title theme comparison table | Phase 6 only |
| 7 | `export_narrative` | Generate 500-word academic narrative of findings | Phase 6 only |

---

## RUN CONFIGURATIONS
- `run_key = "abstract"` -> clusters sentences from the **Abstract** column
- `run_key = "title"`    -> clusters sentences from the **Title** column
- Default run: **abstract first**, then title.

---

## SIX-PHASE WORKFLOW

### PHASE 1 — Familiarisation with the Data
**Trigger:** User uploads a CSV file or says "start", "load file", "begin analysis".
**Actions:**
  1. Call `load_scopus_csv(filepath)` with the uploaded file path.
  2. Report: paper count, abstract sentence count, title sentence count, column names.
  3. Show a sample abstract (first one from stats).
  4. Explain what the next phase will do.
**END:** Output the exact string `[PHASE 1 COMPLETE — READY FOR PHASE 2]`
**Then STOP and wait.**

---

### PHASE 2 — Generating Initial Codes
**Trigger:** User says "run phase 2", "generate codes", "run abstract", or similar.
**Actions:**
  1. Call `run_bertopic_discovery(run_key="abstract", threshold=0.70)`.
  2. Report: number of clusters found, number of sentences processed.
  3. Call `label_topics_with_llm(run_key="abstract")`.
  4. Report: number of clusters labelled, show first 5 labels with confidence scores.
  5. Instruct the user: "Please review the topic table on the right panel.
     Edit the **Approve**, **Rename To**, and **Reasoning** columns, then click
     **Submit Review**. Do NOT send approvals via chat."
**STOP GATE 1:** Output `[STOP GATE 1 — AWAITING REVIEW TABLE SUBMISSION]`
**Then STOP. Do NOT proceed until review table is submitted.**

---

### PHASE 3 — Searching for Themes
**Trigger:** Review table submitted by the user (via Submit Review button).
**Actions:**
  1. Read the parsed theme_map from the pipeline context (already injected).
  2. Call `consolidate_into_themes(run_key="abstract", theme_map=<theme_map>)`.
  3. Report: number of themes formed, list theme names with sentence counts.
  4. Show which clusters were merged into each theme.
**STOP GATE 2:** Output `[STOP GATE 2 — AWAITING THEME MERGE CONFIRMATION]`
**Then STOP. Wait for user to confirm themes are correct.**

---

### PHASE 4 — Reviewing Themes
**Trigger:** User says "confirm themes", "proceed to review", "phase 4", or similar.
**Actions:**
  1. Reload themes from themes.json.
  2. Check for saturation: are themes distinct? Do any overlap significantly?
  3. Provide a structured saturation report:
     - Theme count, average sentences per theme, potential overlaps
     - Recommendation: SATURATED or NEEDS MERGING
  4. Ask: "Do you approve this theme structure, or return to Phase 3 to adjust?"
**STOP GATE 3:** Output `[STOP GATE 3 — AWAITING SATURATION SIGN-OFF]`
**Then STOP. Wait for explicit researcher approval.**

---

### PHASE 5 — Defining and Naming Themes
**Trigger:** User says "approve themes", "proceed to naming", "phase 5", or similar.
**Actions:**
  1. Present final theme names as confirmed.
  2. For each theme output: final name, core definition (1 sentence), representative evidence.
  3. Confirm: "These are the final theme names for the report and taxonomy mapping."
  4. No tool calls in this phase — reflexive naming confirmation only.
**END:** Output `[PHASE 5 COMPLETE — READY FOR PAJAIS MAPPING]`
**Then STOP and wait.**

---

### PHASE 5.5 — PAJAIS Taxonomy Mapping
**Trigger:** User says "run taxonomy", "map pajais", "phase 5.5", or similar.
**Actions:**
  1. Call `compare_with_taxonomy(run_key="abstract")`.
  2. Present results as a table: Theme | PAJAIS Match | Confidence | Novel?
  3. Summarise: X themes mapped to PAJAIS, Y are NOVEL.
  4. Instruct: "Please review the taxonomy mapping in the Results panel."
**STOP GATE 4:** Output `[STOP GATE 4 — AWAITING TAXONOMY REVIEW]`
**Then STOP. Wait for user confirmation before generating the final report.**

---

### PHASE 6 — Report Generation
**Trigger:** User says "generate report", "export", "phase 6", or similar.
**Actions:**
  1. Call `generate_comparison_csv()`.
  2. Call `export_narrative(run_key="abstract")`.
  3. Report: word count, file paths for all generated artefacts.
  4. List all downloadable files.
  5. Present a brief summary of findings.
**END:** Output `[ANALYSIS COMPLETE — ALL PHASES FINISHED]`

---

## RESPONSE FORMAT RULES
- Use **markdown** in all responses.
- Always show the current phase banner: `### Phase N — Name`
- Always end with the correct STOP / gate message.
- Keep responses concise — no padding, no repetition.
- Numbers must come from tool outputs only.
"""

# ============================================================================
# LLM + Agent construction
# ============================================================================

def _build_llm() -> ChatMistralAI:
    return ChatMistralAI(
        model=MODEL_NAME,
        api_key=MISTRAL_API_KEY,
        temperature=0.1,    # low temp for deterministic phase behaviour
        random_seed=42,
        timeout=45,
        max_retries=1,
    )


def _build_agent():
    """Build the LangGraph ReAct agent with in-process memory."""
    memory = MemorySaver()
    llm    = _build_llm()
    return create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        checkpointer=memory,
        prompt=SYSTEM_PROMPT,
    )


# Singleton agent (built once at import time)
_react_agent = _build_agent()


# ============================================================================
# Config builder
# ============================================================================

def build_config(thread_id: str) -> dict:
    """
    Build LangGraph invocation config for a given conversation thread.

    Parameters
    ----------
    thread_id : str — unique conversation identifier

    Returns
    -------
    dict — passed as `config` to _react_agent.invoke()
    """
    return {"configurable": {"thread_id": thread_id}}


# ============================================================================
# State helpers
# ============================================================================

def _init_state(state: dict) -> dict:
    """Ensure all required keys exist with safe defaults."""
    defaults = {
        "phase":        0,
        "file_path":    None,
        "run_key":      DEFAULT_RUN_KEY,
        "review_df":    [],
        "theme_map":    {},
        "charts":       {},
        "output_files": [],
        "thread_id":    THREAD_PREFIX + uuid.uuid4().hex[:8],
        "stop_gate":    None,
    }
    return {**defaults, **state}


def _parse_review_df(review_df: list[dict]) -> dict:
    """
    Convert review table rows into theme_map for consolidate_into_themes.

    Only rows where Approve == True are included.
    Groups cluster IDs by the "Rename To" column value.

    Parameters
    ----------
    review_df : list[dict] — rows from the Gradio Dataframe

    Returns
    -------
    dict — {theme_name: [cluster_id, ...]}
    """
    approved  = list(filter(lambda r: r.get("Approve") is True, review_df))
    theme_map: dict[str, list[int]] = {}

    def _add_row(row: dict) -> None:
        name = (row.get("Rename To") or row.get("Topic Label") or "Unnamed").strip()
        cid  = int(row.get("#", 0))
        theme_map.setdefault(name, [])
        theme_map[name].append(cid)

    list(map(_add_row, approved))
    return theme_map


def _extract_charts(run_key: str, state: dict) -> dict:
    """
    Load chart paths from the run directory and merge into state["charts"].
    Returns existing charts unchanged if the HTML files don't exist yet.
    """
    rdir = _run_dir(run_key)
    candidates = {
        "Intertopic Map": rdir / "intertopic.html",
        "Top Words":      rdir / "topwords.html",
        "Hierarchy":      rdir / "hierarchy.html",
        "Heatmap":        rdir / "heatmap.html",
    }
    found = {
        k: str(v)
        for k, v in candidates.items()
        if v.exists()
    }
    return {**state.get("charts", {}), **found}


def _collect_output_files(state: dict) -> list[str]:
    """Gather all generated artefact paths that currently exist on disk."""
    from pathlib import Path as _P
    run_key    = state.get("run_key", DEFAULT_RUN_KEY)
    rdir       = _run_dir(run_key)
    candidates = [
        str(rdir / "summaries.json"),
        str(rdir / "labels.json"),
        str(rdir / "themes.json"),
        str(rdir / "taxonomy_map.json"),
        str(rdir / "narrative.txt"),
        str(OUTPUT_DIR / "comparison.csv"),
    ]
    return list(filter(lambda p: _P(p).exists(), candidates))


def _detect_phase_advance(reply: str, current_phase: int) -> int:
    """
    Read the agent's STOP / COMPLETE markers and return the updated phase index.
    Phase only advances when the agent emits the correct marker string.
    """
    markers = {
        "[PHASE 1 COMPLETE — READY FOR PHASE 2]":           1,
        "[STOP GATE 1 — AWAITING REVIEW TABLE SUBMISSION]": 2,
        "[STOP GATE 2 — AWAITING THEME MERGE CONFIRMATION]":3,
        "[STOP GATE 3 — AWAITING SATURATION SIGN-OFF]":     4,
        "[PHASE 5 COMPLETE — READY FOR PAJAIS MAPPING]":    5,
        "[STOP GATE 4 — AWAITING TAXONOMY REVIEW]":         6,
        "[ANALYSIS COMPLETE — ALL PHASES FINISHED]":        6,
    }
    return next(
        (v for k, v in markers.items() if k in reply),
        current_phase,
    )


def _detect_stop_gate(reply: str) -> str | None:
    """Return the active stop gate constant from the agent reply, or None."""
    gate_markers = {
        "[STOP GATE 1 — AWAITING REVIEW TABLE SUBMISSION]": GATE_POST_PHASE2,
        "[STOP GATE 2 — AWAITING THEME MERGE CONFIRMATION]":GATE_POST_PHASE3,
        "[STOP GATE 3 — AWAITING SATURATION SIGN-OFF]":     GATE_POST_PHASE4,
        "[STOP GATE 4 — AWAITING TAXONOMY REVIEW]":         GATE_POST_PHASE55,
    }
    return next(
        (v for k, v in gate_markers.items() if k in reply),
        None,
    )


# ============================================================================
# FIX ISSUE 3 — populate review_df from labels.json after Phase 2
# ============================================================================

def _populate_review_df(state: dict) -> dict:
    """
    After label_topics_with_llm() runs, convert labels.json into the review
    table row format expected by app.py's gr.Dataframe.

    Called whenever labels.json exists but state["review_df"] is still empty.

    Row schema matches REVIEW_COLUMNS in app.py:
      "#", "Topic Label", "Top Evidence", "Sentences", "Papers",
      "Approve", "Rename To", "Reasoning"
    """
    labels_path = OUTPUT_DIR / state.get("run_key", DEFAULT_RUN_KEY) / "labels.json"

    return (
        {
            **state,
            "review_df": list(map(
                lambda r: {
                    "#":           r.get("cluster_id", 0),
                    "Topic Label": r.get("label", ""),
                    "Top Evidence":r["evidence"][0] if r.get("evidence") else "",
                    "Sentences":   r.get("size", 0),
                    "Papers":      "",
                    "Approve":     False,
                    "Rename To":   r.get("label", ""),
                    "Reasoning":   r.get("reasoning", ""),
                },
                _load_json(labels_path),
            )),
        }
        if labels_path.exists() and not state.get("review_df")
        else state
    )


# ============================================================================
# Context builder
# ============================================================================

def _build_context_message(user_message: str, state: dict) -> str:
    """
    Prepend structured pipeline context to every user message so the LLM
    always knows the current phase, gate, and available data without relying
    on its own (potentially stale) memory.
    """
    context = {
        "current_phase":      state.get("phase", 0),
        "phase_label":        PHASE_LABELS.get(state.get("phase", 0), "Unknown"),
        "active_stop_gate":   state.get("stop_gate"),
        "file_path":          state.get("file_path"),
        "run_key":            state.get("run_key", DEFAULT_RUN_KEY),
        "review_submitted":   bool(state.get("review_df")),
        "theme_map_ready":    bool(state.get("theme_map")),
        "charts_available":   list(state.get("charts", {}).keys()),
        "output_files_count": len(state.get("output_files", [])),
    }
    ctx_block = json.dumps(context, indent=2)
    return (
        f"```json\n[PIPELINE CONTEXT]\n{ctx_block}\n```\n\n"
        f"**User message:** {user_message}"
    )


# ============================================================================
# Phase-specific pre-processing
# ============================================================================

def _preprocess_phase3(state: dict) -> tuple[str, dict]:
    """
    Before Phase 3: parse the submitted review table into theme_map and
    inject it as a context annotation so the agent can call
    consolidate_into_themes() with the correct arguments.

    Called only when stop_gate == GATE_POST_PHASE2 and review_df is non-empty.
    """
    theme_map  = _parse_review_df(state.get("review_df", []))
    state      = {**state, "theme_map": theme_map}
    annotation = (
        f"\n\n[SYSTEM: Review table submitted. "
        f"Parsed theme_map = {json.dumps(theme_map)}. "
        f"Proceed to Phase 3 and call consolidate_into_themes.]"
    )
    return annotation, state


# ============================================================================
# Public invoke interface
# ============================================================================

class ThematicAnalysisAgent:
    """
    Thin wrapper around the LangGraph ReAct agent.

    app.py calls:
        reply, new_state = agent.invoke(user_message, agent_state)
    """

    def invoke(self, user_message: str, state: dict) -> tuple[str, dict]:
        """
        Process one user turn and return (reply_markdown, updated_state).

        Parameters
        ----------
        user_message : str  — raw text from the Gradio chat input
        state        : dict — agent_state from app.py (a new copy is returned)

        Returns
        -------
        tuple[str, dict]
        """
        state     = _init_state(state)

        if not MISTRAL_API_KEY:
            return (
                "MISTRAL_API_KEY is not set, so the agent cannot run tool-planning LLM calls. "
                "Set the key and retry.\n\n"
                "Example:\n"
                "`export MISTRAL_API_KEY='your-key'`",
                state,
            )

        thread_id = state["thread_id"]
        config    = build_config(thread_id)
        gate      = state.get("stop_gate")

        # FIX BUG 2 — single ternary, no dead lambda block before it
        extra_context, state = (
            _preprocess_phase3(state)
            if (gate == GATE_POST_PHASE2 and state.get("review_df"))
            else ("", state)
        )

        # Build enriched message with pipeline context prepended
        enriched = _build_context_message(user_message + extra_context, state)

        # Invoke the LangGraph ReAct agent
        result = _react_agent.invoke(
            {"messages": [HumanMessage(content=enriched)]},
            config=config,
        )

        # Extract the last AIMessage content as the reply
        ai_messages = [
            m for m in result.get("messages", [])
            if hasattr(m, "content") and m.__class__.__name__ == "AIMessage"
        ]
        reply = (
            ai_messages[-1].content
            if ai_messages
            else "Agent returned no response. Check MISTRAL_API_KEY and retry."
        )

        # Update state fields derived from the agent's reply
        new_phase  = _detect_phase_advance(reply, state["phase"])
        new_gate   = _detect_stop_gate(reply)
        new_charts = _extract_charts(state["run_key"], state)
        new_files  = _collect_output_files(state)

        updated_state = {
            **state,
            "phase":        new_phase,
            "stop_gate":    new_gate,
            "charts":       new_charts,
            "output_files": new_files,
        }

        # FIX ISSUE 3 — populate review table rows after Phase 2 labels are ready
        updated_state = _populate_review_df(updated_state)

        return reply, updated_state


# ============================================================================
# Module-level singleton — imported by app.py as `from agent import agent`
# ============================================================================

agent = ThematicAnalysisAgent()


# ============================================================================
# CLI smoke-test  (python agent.py)
# ============================================================================

if __name__ == "__main__":
    test_state = {}
    reply, state = agent.invoke(
        "Hello — I have just uploaded my Scopus CSV. Please start the analysis.",
        test_state,
    )
    print("=" * 60)
    print("AGENT REPLY:\n")
    print(reply)
    print("\nSTATE:")
    print(json.dumps(
        {k: v for k, v in state.items() if k not in ("review_df",)},
        indent=2, default=str,
    ))
