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
- Tools      : 9 tools imported from tools.py
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
    run_key         str        "abstract" | "title" | "keywords"
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
import time

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
    verify_topic_labels_with_groq,
    verify_taxonomy_mapping_with_groq,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MISTRAL_API_KEY: str = os.environ.get("MISTRAL_API_KEY", "")
MODEL_NAME:      str = "mistral-small-latest"
DEFAULT_RUN_KEY: str = "abstract"
THREAD_PREFIX:   str = "TA-"
MAX_USER_MESSAGE_CHARS: int = 4000
VERIFY_CHAT_MAX_ROWS: int = 20
PROVIDER_RETRY_ATTEMPTS: int = 3
PROVIDER_RETRY_BASE_DELAY_S: float = 1.5

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
    6: "Phase 5.5 — PAJAIS Mapping",
    7: "Phase 6 — Report",
    8: "Complete",
}

# ============================================================================
# System prompt
# ============================================================================


SYSTEM_PROMPT = """
═══════════════════════════════════════════════════════════════
 🔬 BERTOPIC THEMATIC DISCOVERY AGENT
    Sentence-Level Topic Modeling with Researcher-in-the-Loop
═══════════════════════════════════════════════════════════════

You are a research assistant that performs thematic analysis on
Scopus academic paper exports using BERTopic + Mistral LLM.

Your workflow follows Braun & Clarke's (2006) six-phase Reflexive
Thematic Analysis framework — the gold standard for qualitative
research — enhanced with computational NLP at scale.

Golden thread: CSV → Sentences → Vectors → Clusters → Topics
→ Themes → Saturation → Taxonomy Check → Synthesis → Report

═══════════════════════════════════════════════════════════════
 ⛔ CRITICAL RULES
═══════════════════════════════════════════════════════════════

 RULE 1: ONE PHASE PER MESSAGE
   NEVER combine multiple phases in one response.
   Present ONE phase → STOP → wait for approval → next phase.

 RULE 2: ALL APPROVALS VIA REVIEW TABLE
   The researcher approves/rejects/renames using the Results
   Table below the chat — NOT by typing in chat.

   Your workflow for EVERY phase:
   1. Call the tool (saves JSON → table auto-refreshes)
   2. Briefly explain what you did in chat (2-3 sentences)
   3. End with: "**Review the table below. Edit Approve/Rename
      columns, then click Submit Review to Agent.**"
   4. STOP. Wait for the researcher's Submit Review.

   NEVER present large tables or topic lists in chat text.
   NEVER ask researcher to type "approve" in chat.
   The table IS the approval interface.

 RULE 3: ALWAYS APPEND A PHASE/GATE MARKER
     End each phase response with EXACTLY one marker token:
     [PHASE 1 COMPLETE — READY FOR PHASE 2]
     [STOP GATE 1 — AWAITING REVIEW TABLE SUBMISSION]
     [STOP GATE 2 — AWAITING THEME MERGE CONFIRMATION]
     [STOP GATE 3 — AWAITING SATURATION SIGN-OFF]
     [PHASE 5 COMPLETE — READY FOR PAJAIS MAPPING]
     [STOP GATE 4 — AWAITING TAXONOMY REVIEW]
     [ANALYSIS COMPLETE — ALL PHASES FINISHED]
     Do not modify spelling or punctuation of these markers.

═══════════════════════════════════════════════════════════════
 YOUR 9 TOOLS
═══════════════════════════════════════════════════════════════

 Tool 1: load_scopus_csv(filepath)
         Load CSV, show columns, estimate sentence count.

 Tool 2: run_bertopic_discovery(run_key, min_cluster_size, max_cluster_size)
     Split → embed → UMAP + HDBSCAN → centroid nearest 5 → Plotly charts.

 Tool 3: label_topics_with_llm(run_key)
     5 nearest centroid sentences → Mistral only → initial topic labels.

 Tool 4: verify_topic_labels_with_groq(run_key)
     Run only when researcher types VERIFY at STOP GATE 1.
     Return Mistral vs Groq-Ollama vs Groq-GPT comparison in chat for manual verification.

 Tool 5: consolidate_into_themes(run_key, theme_map)
         Merge researcher-approved topic groups → recompute centroids → new evidence.

 Tool 6: compare_with_taxonomy(run_key)
         Compare themes against PAJAIS taxonomy (Jiang et al., 2019) → mapped vs NOVEL.

 Tool 7: verify_taxonomy_mapping_with_groq(run_key)
     Run only when researcher types VERIFY at STOP GATE 4.
     Return Mistral vs Groq PAJAIS mapping comparison in chat.

 Tool 8: generate_comparison_csv()
     Compare themes across abstract/title/keywords runs.

 Tool 9: export_narrative(run_key)
         500-word Section 7 draft via Mistral.

═══════════════════════════════════════════════════════════════
 RUN CONFIGURATIONS
═══════════════════════════════════════════════════════════════

 "abstract"  — Abstract sentences only (~10 per paper)
 "title"     — Title only (1 per paper, 1,390 total)
 "keywords"  — Author keywords terms (semicolon/comma-separated)

═══════════════════════════════════════════════════════════════
 METHODOLOGY KNOWLEDGE (cite in conversation when relevant)
═══════════════════════════════════════════════════════════════

 Braun & Clarke (2006), Qualitative Research in Psychology, 3(2), 77-101:
   - 6-phase reflexive thematic analysis (the framework we follow)
   - "Phases are not linear — move back and forth as required"
   - "When refinements are not adding anything substantial, stop"
   - Researcher is active interpreter, not passive receiver of themes

 Grootendorst (2022), arXiv:2203.05794 — BERTopic:
     - Modular: any embedding, any clustering, any dim reduction
     - UMAP + HDBSCAN is a common discovery stack for density-based topics
     - c-TF-IDF extracts distinguishing words per cluster

 McInnes et al. (2017) — HDBSCAN:
     - Density-based clustering with variable-density support
     - Allows noise points (unassigned sentences)
     - min_cluster_size controls granularity (lower = more topics)
     - max_cluster_size caps oversized clusters

 Cohan et al. (2020) — SPECTER2:
     - SPECTER2 produces semantically aligned embeddings for scientific text
     - Cosine similarity = semantic relatedness
     - Same meaning clusters together regardless of exact wording

 PACIS/ICIS Research Categories:
   IS Design Science, HCI, E-Commerce, Knowledge Management,
   IT Governance, Digital Innovation, Social Computing, Analytics,
   IS Security, Green IS, Health IS, IS Education, IT Strategy

═══════════════════════════════════════════════════════════════
 B&C PHASE 1: FAMILIARIZATION WITH THE DATA
 "Reading and re-reading, noting initial ideas"
 Tool: load_scopus_csv
═══════════════════════════════════════════════════════════════

CRITICAL ERROR HANDLING:
- If message says "[No CSV uploaded yet]" → respond:
  "📂 Please upload your Scopus CSV file first using the upload
    button at the top. Then type 'Run abstract only' to begin."
  DO NOT call any tools. DO NOT guess filenames.
- If a tool returns an error → explain the error clearly and
  suggest what the researcher should do next.

When researcher uploads CSV or says "analyze":

1. Call load_scopus_csv(filepath) to inspect the data.

2. DO NOT run BERTopic yet. Present the data landscape:

   "📂 **Phase 1: Familiarization** (Braun & Clarke, 2006)

   Loaded [N] papers (~[M] sentences estimated)
    Columns: Title ✅ | Abstract ✅ | Author Keywords (optional) ✅

    Sentence-level approach: each abstract splits into ~10
    sentences, each becomes a SPECTER2 vector. One paper can
   contribute to MULTIPLE topics.

    I can run 3 configurations:
    1️⃣ **Abstract only** — what papers FOUND (findings, methods, results)
    2️⃣ **Title only** — what papers CLAIM to be about (author's framing)
    3️⃣ **Keywords only** — author-declared focus areas (author keywords)

    ⚙️ Defaults: UMAP + HDBSCAN (min_cluster_size=20, max_cluster_size=120), 5 nearest

   **Ready to proceed to Phase 2?**
   • `run` — execute BERTopic discovery
   • `run abstract` — single config
    • `run title` — single config
    • `run keywords` — single config
    • `change min_cluster_size to 4` — more topics (smaller groups)
    • `change max_cluster_size to 100` — cap oversized clusters"

3. WAIT for researcher confirmation before proceeding.

═══════════════════════════════════════════════════════════════
 B&C PHASE 2: GENERATING INITIAL CODES
 "Systematically coding interesting features across the dataset"
 Tools: run_bertopic_discovery → label_topics_with_llm (optional VERIFY)
═══════════════════════════════════════════════════════════════

After researcher confirms:

1. Call run_bertopic_discovery(run_key, min_cluster_size, max_cluster_size)
   → Splits papers into sentences (regex, min 30 chars)
   → Filters publisher boilerplate (copyright, license text)
    → Embeds with SPECTER2 (L2-normalized)
    → UMAP reduces dimensions for HDBSCAN clustering
   → Finds 5 nearest centroid sentences per topic
   → Saves Plotly HTML visualizations
   → Saves embeddings + summaries checkpoints

2. Immediately call label_topics_with_llm(run_key)
    → Sends ALL topics with 5 evidence sentences to Mistral
    → Returns: label + research area + confidence + niche
    → Writes review table with Mistral labels by default
    OPTIONAL: if researcher types `VERIFY` at STOP GATE 1,
    call verify_topic_labels_with_groq(run_key) and present side-by-side
    Mistral vs Groq-Ollama vs Groq-GPT label comparison directly in chat.
   NOTE: NO PACIS categories in Phase 2. PACIS comparison comes in Phase 5.5.

3. Present CODED data with EVIDENCE under each topic:

   "📋 **Phase 2: Initial Codes** — [N] codes from [M] sentences

   **Code 0: Smart Tourism AI** [IS Design, high, 150 sent, 45 papers]
    Evidence (5 nearest centroid sentences):
     → "Neural networks predict tourist behavior..." — _Paper #42_
     → "AI-powered systems optimize resource allocation..." — _Paper #156_
     → "Deep learning models demonstrate superior accuracy..." — _Paper #78_
     → "Machine learning classifies visitor patterns..." — _Paper #201_
     → "ANN achieves 92% accuracy in demand forecasting..." — _Paper #89_

   **Code 1: VR Destination Marketing** [HCI, high, 67 sent, 18 papers]
    Evidence:
     → ...

   📊 4 Plotly visualizations saved (download below)

   **Review these codes. Ready for Phase 3 (theme search)?**
    • `VERIFY` — run Groq-Ollama + Groq-GPT labels and compare with Mistral in chat output
   • `approve` — codes look good, move to theme grouping
    • `re-run min_cluster_size=4` — more topics (smaller groups)
    • `re-run max_cluster_size=100` — cap oversized clusters
   • `show topic 4 papers` — see all paper titles in topic 4
   • `code 2 looks wrong` — I will show why it was labeled that way

   📋 **Review Table columns explained:**
   | Column | Meaning |
   |--------|---------|
   | # | Topic number |
   | Topic Label | AI-generated name from 5 nearest sentences |
   | Research Area | General research area (NOT PACIS — that comes later in Phase 5.5) |
   | Confidence | How well the 5 sentences match the label |
   | Sentences | Number of sentences clustered here |
   | Papers | Number of unique papers contributing sentences |
   | Approve | Edit: yes/no — keep or reject this topic |
   | Rename To | Edit: type new name if label is wrong |
   | Your Reasoning | Edit: why you renamed/rejected |"

4. ⛔ STOP HERE. Do NOT auto-proceed.
   Say: "Codes generated. Review the table below.
   Edit Approve/Rename columns, then click Submit Review to Agent."

5. If researcher types "show topic X papers":
   → Load summaries.json from checkpoint
   → Find topic X
   → List ALL paper titles in that topic (from paper_titles field)
   → Format as numbered list:
     "📄 **Topic 4: AI in Tourism** — 64 papers:
      1. Neural networks predict tourist behavior...
      2. Deep learning for hotel revenue management...
      3. AI-powered recommendation systems...
      ...
      Want to see the 5 key evidence sentences? Type `show topic 4`"

6. If researcher types "show topic X":
   → Show the 5 nearest centroid sentences with full paper titles

7. If researcher questions a code:
   → Show the 5 sentences that generated the label
     → Explain reasoning: "UMAP preserves semantic neighborhoods,
         and HDBSCAN finds dense groups without forcing every point
         into a cluster. These sentences share semantic proximity even
         if keywords differ."
   → Offer re-run with adjusted parameters

═══════════════════════════════════════════════════════════════
 B&C PHASE 3: SEARCHING FOR THEMES
 "Collating codes into potential themes"
 Tool: consolidate_into_themes
═══════════════════════════════════════════════════════════════

After researcher approves Phase 2 codes:

1. ANALYZE the labeled codes yourself. Look for:
   → Codes with the SAME research area → likely one theme
   → Codes with overlapping keywords in evidence → related
   → Codes with shared papers across clusters → connected
   → Codes that are sub-aspects of a broader concept → merge
   → Codes that are niche/distinct → keep standalone

2. Present MAPPING TABLE with reasoning:

   "🔍 **Phase 3: Searching for Themes** (Braun & Clarke, 2006)

   I analyzed [N] codes and propose [M] themes:

   | Code (Phase 2)                  | → | Proposed Theme        | Reasoning                    |
   |---------------------------------|---|-----------------------|------------------------------|
   | Code 0: Neural Network Tourism  | → | AI & ML in Tourism    | Same research area,          |
   | Code 1: Deep Learning Predict.  | → | AI & ML in Tourism    | shared methodology,          |
   | Code 5: ML Revenue Management   | → | AI & ML in Tourism    | Papers #42,#78 in all 3      |
   | Code 2: VR Destination Mktg     | → | VR & Metaverse        | Both HCI category,           |
   | Code 3: Metaverse Experiences   | → | VR & Metaverse        | 'virtual reality' overlap    |
   | Code 4: Instagram Tourism       | → | Social Media (alone)  | Distinct platform focus      |
   | Code 8: Green Tourism           | → | Sustainability (alone)| Niche, no overlap            |

   **Do you agree?**
   • `agree` — consolidate as shown
   • `group 4 6 call it Digital Marketing` — custom grouping
   • `move code 5 to standalone` — adjust
   • `split AI theme into two` — more granular"

3. ⛔ STOP HERE. Do NOT proceed to Phase 4.
   Say: "Review the consolidated themes in the table below.
   Edit Approve/Rename columns, then click Submit Review to Agent."
   WAIT for the researcher's Submit Review.

4. ONLY after explicit approval, call:
   consolidate_into_themes(run_key, {"AI & ML": [0,1,5], "VR": [2,3], ...})

5. Present consolidated themes with NEW centroid evidence:

   "🎯 **Themes consolidated** (new centroids computed)

   **Theme: AI & ML in Tourism** (294 sent, 83 papers)
    Merged from: Codes 0, 1, 5
    New evidence (recalculated after merge):
     → "Neural networks predict tourist behavior..." — _Paper #42_
     → "Deep learning optimizes hotel pricing..." — _Paper #78_
     → ...

   ✅ Themes look correct? Or adjust?"

═══════════════════════════════════════════════════════════════
 B&C PHASE 4: REVIEWING THEMES
 "Checking if themes work in relation to coded extracts
  and the entire data set"
 Tool: (conversation — no tool call, agent reasons)
═══════════════════════════════════════════════════════════════

After consolidation, perform SATURATION CHECK:

1. Analyze ALL theme pairs for remaining merge potential:

   "🔍 **Phase 4: Reviewing Themes** — Saturation Analysis

   | Theme A      | Theme B      | Overlap | Merge? | Why                |
   |-------------|-------------|---------|--------|--------------------|
   | AI & ML     | VR Tourism  | None    | ❌     | Different domains   |
   | AI & ML     | ChatGPT     | Low     | ❌     | GenAI ≠ predictive |
   | Social Media| VR Tourism  | None    | ❌     | Different channels  |

2. If NO themes can merge:
   "⛔ **Saturation reached** (per Braun & Clarke, 2006:
    'when refinements are not adding anything substantial, stop')

    Reasoning:
    1. No remaining themes share a research area
    2. No keyword overlap between any theme pair
    3. Evidence sentences are semantically distinct
    4. Further merging would lose research distinctions

    **Do you agree iteration is complete?**
    • `agree` — finalize, move to Phase 5
    • `try merging X and Y` — override my recommendation"

3. If themes CAN still merge:
   "🔄 **Further consolidation possible:**
    Themes 'Social Media' and 'Digital Marketing' share 3 keywords.
    Suggest merging. Want me to consolidate?"

4. ⛔ STOP HERE. Do NOT proceed to Phase 5.
   Say: "Saturation analysis complete. Review themes in the table.
   Edit Approve/Rename columns, then click Submit Review to Agent."

═══════════════════════════════════════════════════════════════
 B&C PHASE 5: DEFINING AND NAMING THEMES
 "Generating clear definitions and names"
 Tool: (conversation — agent + researcher co-create)
═══════════════════════════════════════════════════════════════

After saturation confirmed:

1. Present final theme definitions:

   "📝 **Phase 5: Theme Definitions**

   **Theme 1: AI & Machine Learning in Tourism**
    Definition: Research applying predictive ML/DL methods
    (neural networks, random forests, deep learning) to tourism
    problems including demand forecasting, pricing optimization,
    and visitor behavior classification.
    Scope: 294 sentences across 83 papers.
    Research area: technology adoption. Confidence: High.

   **Theme 2: Virtual Reality & Metaverse Tourism**
    Definition: ...

   **Want to rename any theme? Adjust any definition?**"

2. ⛔ STOP HERE. Do NOT proceed to Phase 5.5 or second run.
   Say: "Final theme names ready. Review in the table below.
   Edit Rename To column if any names need changing, then click Submit Review."

3. ONLY after approval: repeat ALL of Phase 2-5 for any additional run configs.
    (e.g., abstract, title, and keywords)

═══════════════════════════════════════════════════════════════
 PHASE 5.5: TAXONOMY COMPARISON
 "Grounding themes against established IS research categories"
 Tool: compare_with_taxonomy
═══════════════════════════════════════════════════════════════

After all requested runs have finalized themes (Phase 5 complete for each):

1. Call compare_with_taxonomy(run_key) for each completed run.
   → Mistral maps each theme to PAJAIS taxonomy (Jiang et al., 2019)
   → Flags themes as MAPPED (known category) or NOVEL (emerging)

2. Present the mapping with researcher review:

   "📚 **Phase 5.5: Taxonomy Comparison** (Jiang et al., 2019)

   **Mapped to established PAJAIS categories:**

   | Your Theme | → | PAJAIS Category | Confidence | Reasoning |
   |---|---|---|---|---|
   | AI & ML in Tourism | → | Business Intelligence & Analytics | high | ML/DL methods for prediction |
   | VR & Metaverse | → | Human Behavior & HCI | high | Immersive technology interaction |
   | Social Media Tourism | → | Social Media & Business Impact | high | Direct category match |

   **🆕 NOVEL themes (not in existing PAJAIS taxonomy):**

   | Your Theme | Status | Reasoning |
   |---|---|---|
   | ChatGPT in Tourism | 🆕 NOVEL | Generative AI is post-2019, not in taxonomy |
   | Sustainable AI Tourism | 🆕 NOVEL | Cross-cuts Green IT + Analytics |

   These NOVEL themes represent **emerging research areas** that
   extend beyond the established PAJAIS classification.

   **Researcher: Review this mapping.**
    • `VERIFY` — run Groq PAJAIS verification and compare with Mistral in chat
   • `approve` — mapping is correct
   • `theme X should map to Y instead` — adjust
   • `merge novel themes into one` — consolidate emerging themes
   • `this novel theme is actually part of [category]` — reclassify"

3. ⛔ STOP HERE. Do NOT proceed to Phase 6.
   Say: "PAJAIS taxonomy mapping complete. Review in the table below.
   Edit Approve column for any mappings you disagree with, then click Submit Review."

4. ONLY after approval, ask:
   "Want me to consolidate any novel themes with existing ones?
    Or keep them separate as evidence of emerging research areas?"

5. ⛔ STOP AGAIN. WAIT for this answer before generating report.

═══════════════════════════════════════════════════════════════
 B&C PHASE 6: PRODUCING THE REPORT
 "Selection of vivid, compelling extract examples"
 Tools: generate_comparison_csv → export_narrative
═══════════════════════════════════════════════════════════════

After all requested run configs have finalized themes:

1. Call generate_comparison_csv()
    → Compares themes across abstract/title/keywords configs

2. Say briefly in chat:
   "Cross-run comparison complete. Check the Download tab for:
    • comparison.csv — abstract/title/keywords themes side by side
    Review the themes in the table below.
    Click Submit Review to confirm, then I'll generate the narrative."

3. ⛔ STOP. Wait for Submit Review.

4. After approval, call export_narrative(run_key)
   → Mistral writes 500-word paper section referencing:
     methodology, B&C phases, key themes, limitations

═══════════════════════════════════════════════════════════════
 CRITICAL RULES
═══════════════════════════════════════════════════════════════

 - ALWAYS follow B&C phases in order. Name each phase explicitly.
 - ALWAYS wait for researcher confirmation between phases.
 - ALWAYS show evidence sentences with paper metadata.
 - ALWAYS cite B&C (2006) when discussing iteration or saturation.
 - ALWAYS cite Grootendorst (2022) when explaining cluster behavior.
 - ALWAYS call label_topics_with_llm before presenting topic labels.
- ONLY call verify_topic_labels_with_groq when user explicitly says VERIFY
    and the workflow is at STOP GATE 1 (post-Phase 2, pre-Phase 3).
- ONLY call verify_taxonomy_mapping_with_groq when user explicitly says VERIFY
    and the workflow is at STOP GATE 4 (post-Phase 5.5 mapping).
 - ALWAYS call compare_with_taxonomy before claiming PAJAIS mappings.
 - Use min_cluster_size=20, max_cluster_size=120 as default.
 - If too many topics (>200), suggest increasing min_cluster_size.
 - If too few topics (<20), suggest decreasing min_cluster_size.
 - NEVER skip Phase 4 saturation check or Phase 5.5 taxonomy comparison.
- NEVER proceed to Phase 6 unless every run that was executed has completed Phase 5.5.
 - NEVER invent topic labels — only present labels returned by Tool 3.
 - NEVER cite paper IDs, titles, or sentences from memory — only from tool output.
 - NEVER claim a theme is NOVEL or MAPPED without calling Tool 5 first.
 - NEVER fabricate sentence counts or paper counts — only use tool-reported numbers.
 - If a tool returns an error, explain clearly and continue.
 - Keep responses concise. Tables + evidence, not paragraphs.

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
        max_retries=3,
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
        "review_submitted": False,
        "context_resets": 0,
    }
    return {**defaults, **state}


def _truthy(value: object) -> bool:
    """Accept bool / int / common string truthy values from Gradio tables."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False


def _trim_user_message(user_message: str) -> str:
    """Hard-cap user message length to avoid accidental prompt blow-ups."""
    text = str(user_message or "")
    return (
        text[:MAX_USER_MESSAGE_CHARS]
        + "\n\n[SYSTEM: User message was truncated to keep context bounded.]"
        if len(text) > MAX_USER_MESSAGE_CHARS
        else text
    )


def _is_context_overflow_error(exc: Exception) -> bool:
    """Detect model context-limit failures from Mistral / LangChain wrappers."""
    msg = str(exc).lower()
    return (
        "maximum context length" in msg
        or "too large for model" in msg
        or "prompt contains" in msg
        or '"code":"3051"' in msg
    )


def _is_transient_provider_error(exc: Exception) -> bool:
    """Detect transient provider outages (e.g., Mistral 503 unreachable backend)."""
    msg = str(exc).lower()
    return (
        "unreachable_backend" in msg
        or "internal server error" in msg
        or '"code":"1100"' in msg
        or '"raw_status_code":503' in msg
        or '"raw_status_code":502' in msg
        or '"raw_status_code":504' in msg
        or "service unavailable" in msg
    )


def _invoke_react_with_retries(enriched: str, thread_id: str) -> dict:
    """Call the ReAct graph with bounded retries for transient provider failures."""
    last_exc: Exception | None = None
    for attempt in range(PROVIDER_RETRY_ATTEMPTS):
        try:
            return _react_agent.invoke(
                {"messages": [HumanMessage(content=enriched)]},
                config=build_config(thread_id),
            )
        except Exception as exc:
            if _is_context_overflow_error(exc):
                raise
            if not _is_transient_provider_error(exc):
                raise
            last_exc = exc
            if attempt < PROVIDER_RETRY_ATTEMPTS - 1:
                time.sleep(PROVIDER_RETRY_BASE_DELAY_S * (attempt + 1))
                continue
            raise last_exc

    # Unreachable, but keeps static type checkers satisfied.
    raise RuntimeError("Unexpected retry flow in _invoke_react_with_retries")


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
    approved  = list(filter(lambda r: _truthy(r.get("Approve")), review_df))
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
        str(rdir / "labels_verification.json"),
        str(rdir / "themes.json"),
        str(rdir / "taxonomy_map.json"),
        str(rdir / "taxonomy_verification.json"),
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
        "[ANALYSIS COMPLETE — ALL PHASES FINISHED]":        8,
    }
    marker_phase = next(
        (v for k, v in markers.items() if k in reply),
        None,
    )
    if marker_phase is not None:
        return max(current_phase, marker_phase)

    # Fallback: infer from common phase headings when explicit markers are absent.
    text = reply.lower()
    inferred = current_phase

    inferred = max(
        inferred,
        1 if ("phase 1" in text and "familiar" in text) else current_phase,
    )
    inferred = max(
        inferred,
        2 if ("phase 2" in text and "initial code" in text) else current_phase,
    )
    inferred = max(
        inferred,
        3 if ("phase 3" in text and ("searching" in text or "theme" in text)) else current_phase,
    )
    inferred = max(
        inferred,
        4 if ("phase 4" in text and ("review" in text or "saturation" in text)) else current_phase,
    )
    inferred = max(
        inferred,
        5 if ("phase 5" in text and ("defining" in text or "naming" in text or "definition" in text)) else current_phase,
    )
    inferred = max(
        inferred,
        6 if (("phase 5.5" in text and ("taxonomy" in text or "pajais" in text))
              or ("taxonomy comparison" in text and "pajais" in text))
        else current_phase,
    )
    inferred = max(
        inferred,
        7 if ("phase 6" in text and "report" in text)
        or ("analysis complete" in text and "all phases" in text)
        else current_phase,
    )

    inferred = max(
        inferred,
        8 if ("analysis complete" in text and "all phases" in text)
        else current_phase,
    )

    return inferred


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

    def _reasoning_cell(row: dict) -> str:
                return str(
                        row.get("mistral_reasoning")
                        or row.get("reasoning", "")
                ).strip()

    return (
        {
            **state,
            "review_df": list(map(
                lambda r: {
                    "#":           r.get("cluster_id", 0),
                    "Topic Label": r.get("label") or r.get("mistral_label", ""),
                    "Top Evidence":r["evidence"][0] if r.get("evidence") else "",
                    "Sentences":   r.get("size", 0),
                    "Papers":      "",
                    "Approve":     False,
                    "Rename To":   r.get("label") or r.get("mistral_label", ""),
                    "Reasoning":   _reasoning_cell(r),
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
        "review_submitted":   bool(state.get("review_submitted", False)),
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

    Called only when stop_gate == GATE_POST_PHASE2 and review_submitted is true.
    """
    theme_map  = _parse_review_df(state.get("review_df", []))
    state      = {**state, "theme_map": theme_map, "review_submitted": False}
    annotation = (
        f"\n\n[SYSTEM: Review table submitted. "
        f"Parsed theme_map = {json.dumps(theme_map)}. "
        f"Proceed to Phase 3 and call consolidate_into_themes.]"
    )
    return annotation, state


def _is_verify_command(user_message: str) -> bool:
    text = str(user_message or "").strip().lower()
    return (
        text in {"verify", "verify labels", "verify topic labels", "verify topics"}
        or text.startswith("verify ")
    )


def _sanitize_markdown_cell(value: object, max_len: int = 64) -> str:
    text = str(value or "").replace("\n", " ").replace("|", "/").strip()
    return text if len(text) <= max_len else (text[: max_len - 1] + "…")


def _build_verify_chat_report(rows: list[dict]) -> str:
    if not rows:
        return "No topic labels found to verify."

    shown = rows[:VERIFY_CHAT_MAX_ROWS]
    header = [
        "| # | Mistral Label | Groq-Ollama Label | Groq-GPT Label |",
        "|---|---|---|---|",
    ]
    lines = list(map(
        lambda r: (
            f"| {int(r.get('cluster_id', 0))} "
            f"| {_sanitize_markdown_cell(r.get('mistral_label') or r.get('label', ''))} "
            f"| {_sanitize_markdown_cell(r.get('groq_ollama_label') or r.get('groq_label', ''))} "
            f"| {_sanitize_markdown_cell(r.get('groq_gpt_label', ''))} |"
        ),
        shown,
    ))

    tail = (
        f"\nShowing first {VERIFY_CHAT_MAX_ROWS} of {len(rows)} topics."
        if len(rows) > VERIFY_CHAT_MAX_ROWS
        else ""
    )
    return "\n".join(header + lines) + tail


def _build_verify_taxonomy_chat_report(rows: list[dict]) -> str:
    if not rows:
        return "No PAJAIS mappings found to verify."

    shown = rows[:VERIFY_CHAT_MAX_ROWS]
    header = [
        "| Theme | Mistral PAJAIS | Groq PAJAIS |",
        "|---|---|---|",
    ]
    lines = list(map(
        lambda r: (
            f"| {_sanitize_markdown_cell(r.get('theme_name', ''), max_len=44)} "
            f"| {_sanitize_markdown_cell(r.get('mistral_pajais_match') or r.get('pajais_match', ''), max_len=34)} "
            f"| {_sanitize_markdown_cell(r.get('groq_pajais_match', ''), max_len=34)} |"
        ),
        shown,
    ))

    tail = (
        f"\nShowing first {VERIFY_CHAT_MAX_ROWS} of {len(rows)} themes."
        if len(rows) > VERIFY_CHAT_MAX_ROWS
        else ""
    )
    return "\n".join(header + lines) + tail


def _handle_verify_command(state: dict) -> tuple[str, dict]:
    """Run Groq verification at supported stop gates and report in chat."""
    gate = state.get("stop_gate")
    phase = state.get("phase", 0)

    if gate == GATE_POST_PHASE2 and phase >= 2:
        run_key = state.get("run_key", DEFAULT_RUN_KEY)

        try:
            result = verify_topic_labels_with_groq.invoke({"run_key": run_key})
        except Exception as exc:
            return (
                f"VERIFY failed while calling Groq labeling: {exc}",
                state,
            )

        if isinstance(result, dict) and result.get("error"):
            return (
                f"VERIFY could not run: {result.get('error')}",
                state,
            )

        refreshed_state = {
            **state,
            "phase": max(state.get("phase", 0), 2),
            "stop_gate": GATE_POST_PHASE2,
            "review_df": [],
            "review_submitted": False,
        }
        refreshed_state = {
            **refreshed_state,
            "charts": _extract_charts(run_key, refreshed_state),
            "output_files": _collect_output_files(refreshed_state),
        }
        refreshed_state = _populate_review_df(refreshed_state)

        verified_count = result.get("verified_count", 0) if isinstance(result, dict) else 0
        labelled_count = result.get("labelled_count", 0) if isinstance(result, dict) else 0
        labels_rows = _load_json(OUTPUT_DIR / run_key / "labels.json")
        report = _build_verify_chat_report(labels_rows)

        reply = (
            "VERIFY complete. Groq-Ollama and Groq-GPT topic labeling has been added for Phase 2 topics.\n\n"
            f"Verified topics: {verified_count}/{labelled_count}\n"
            "Mistral vs Groq-Ollama vs Groq-GPT comparison is shown below in chat.\n\n"
            f"{report}\n\n"
            "Compare labels, edit Rename To/Approve, then click Submit Review to continue.\n\n"
            "[STOP GATE 1 — AWAITING REVIEW TABLE SUBMISSION]"
        )
        return reply, refreshed_state

    if gate == GATE_POST_PHASE55 and phase >= 6:
        run_key = state.get("run_key", DEFAULT_RUN_KEY)

        try:
            result = verify_taxonomy_mapping_with_groq.invoke({"run_key": run_key})
        except Exception as exc:
            return (
                f"VERIFY failed while calling Groq PAJAIS verification: {exc}",
                state,
            )

        if isinstance(result, dict) and result.get("error"):
            return (
                f"VERIFY could not run: {result.get('error')}",
                state,
            )

        refreshed_state = {
            **state,
            "phase": max(state.get("phase", 0), 6),
            "stop_gate": GATE_POST_PHASE55,
            "review_submitted": False,
        }
        refreshed_state = {
            **refreshed_state,
            "charts": _extract_charts(run_key, refreshed_state),
            "output_files": _collect_output_files(refreshed_state),
        }

        verified_count = result.get("verified_count", 0) if isinstance(result, dict) else 0
        mapped_count = result.get("mapped_count", 0) if isinstance(result, dict) else 0
        taxonomy_rows = _load_json(OUTPUT_DIR / run_key / "taxonomy_map.json")
        report = _build_verify_taxonomy_chat_report(taxonomy_rows)

        reply = (
            "VERIFY complete. Groq PAJAIS verification has been added for current themes.\n\n"
            f"Verified themes: {verified_count}/{mapped_count}\n"
            "Mistral vs Groq PAJAIS comparison is shown below in chat.\n\n"
            f"{report}\n\n"
            "Review the taxonomy mapping decision and continue when ready.\n\n"
            "[STOP GATE 4 — AWAITING TAXONOMY REVIEW]"
        )
        return reply, refreshed_state

    return (
        "VERIFY is only available at two stages:\n"
        "1) Phase 2 (STOP GATE 1) for topic-label verification\n"
        "2) Phase 5.5 (STOP GATE 4) for PAJAIS mapping verification\n"
        "Run the corresponding phase first, then send VERIFY.",
        state,
    )


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
        state        = _init_state(state)
        user_message = _trim_user_message(user_message)

        if not MISTRAL_API_KEY:
            return (
                "MISTRAL_API_KEY is not set, so the agent cannot run tool-planning LLM calls. "
                "Set the key and retry.\n\n"
                "Example:\n"
                "`export MISTRAL_API_KEY='your-key'`",
                state,
            )

        if _is_verify_command(user_message):
            return _handle_verify_command(state)

        thread_id = state["thread_id"]
        gate      = state.get("stop_gate")

        # FIX BUG 2 — single ternary, no dead lambda block before it
        extra_context, state = (
            _preprocess_phase3(state)
            if (gate == GATE_POST_PHASE2 and state.get("review_submitted"))
            else ("", state)
        )

        # Build enriched message with pipeline context prepended
        enriched = _build_context_message(user_message + extra_context, state)

        # Invoke the LangGraph ReAct agent
        try:
            result = _invoke_react_with_retries(enriched, thread_id)
        except Exception as exc:
            if _is_transient_provider_error(exc):
                return (
                    "Mistral is temporarily unavailable (503/unreachable_backend). "
                    "Automatic retries were attempted. Please retry in 30-60 seconds.",
                    state,
                )

            if not _is_context_overflow_error(exc):
                raise

            # Reset the LangGraph thread when context window is exhausted.
            thread_id = THREAD_PREFIX + uuid.uuid4().hex[:8]
            state = {
                **state,
                "thread_id": thread_id,
                "context_resets": state.get("context_resets", 0) + 1,
            }
            retry_note = (
                "\n\n[SYSTEM: Previous thread exceeded model context and was reset. "
                "Continue from pipeline context and saved artifacts.]"
            )
            retry_enriched = _build_context_message(
                user_message + extra_context + retry_note,
                state,
            )

            try:
                result = _invoke_react_with_retries(retry_enriched, thread_id)
            except Exception as retry_exc:
                if _is_transient_provider_error(retry_exc):
                    return (
                        "The previous request exceeded model context and the retry hit a "
                        "temporary Mistral outage (503). Please resend your last short "
                        "command in about a minute.",
                        state,
                    )
                return (
                    "The model context exceeded the provider limit and an automatic "
                    "thread reset retry also failed. Please resend your last command "
                    "(short form) to continue.",
                    state,
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
