"""
tools.py — BERTopic Thematic Analysis Pipeline Tools
=====================================================
Eight LangChain @tool functions implementing Braun & Clarke's (2006)
six-phase thematic analysis pipeline.

Conventions
-----------
- All tools accept / return plain Python dicts (JSON-serialisable).
- Artefacts are written to  OUTPUT_DIR / run_key / <file>.
- Functional style throughout: map, operator, numpy vectorised ops.
- No for/while loops, no try/except, no if/else.

Fixes applied (v2)
------------------
- BUG 1  : run_bertopic_discovery() now saves sent_labels.npy —
           per-sentence cluster-label array required by Tool 4.
- BUG 1  : consolidate_into_themes() _build_theme() rewritten —
           centroid computed from actual merged-cluster embeddings
           via sent_labels.npy mask (no dead `if False` scaffolding).
- ISSUE 1: generate_comparison_csv() guards against missing title run
           with a .exists() check instead of hard-crashing.

Dependencies
------------
    pip install langchain langchain-core langchain-mistralai
                sentence-transformers scikit-learn plotly pandas numpy
"""

# ---------------------------------------------------------------------------
# Stdlib
# ---------------------------------------------------------------------------
import json
import os
import re
import time
from functools import reduce
from pathlib import Path
from operator import itemgetter

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_mistralai import ChatMistralAI

try:
    from langchain_groq import ChatGroq  # type: ignore[import-not-found]
except ImportError:
    ChatGroq = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MISTRAL_API_KEY: str   = os.environ.get("MISTRAL_API_KEY", "")
MODEL_NAME:      str   = "mistral-small-latest"
GROQ_API_KEY: str      = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL_NAME: str   = os.environ.get("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
EMBED_MODEL:     str   = "all-MiniLM-L6-v2"
BASE_DIR:        Path  = Path(__file__).resolve().parent
OUTPUT_DIR:      Path  = BASE_DIR / "outputs"
N_EVIDENCE:      int   = 5       # sentences kept per cluster centroid
DISTANCE_THRESH: float = 0.35   # cosine-distance threshold (1 - similarity)
RANDOM_SEED:     int   = 42
LLM_TIMEOUT_S:   int   = 45
LLM_MAX_RETRIES: int   = 3
MAX_LABEL_CLUSTERS: int = 60
MIN_CLUSTER_SIZE_FOR_LABEL: int = 3
MAX_TOOL_RETURN_PREVIEW: int = 12
PROVIDER_RETRY_ATTEMPTS: int = 3
PROVIDER_RETRY_BASE_DELAY_S: float = 1.5

# Run configurations — keys map to source columns
RUN_CONFIGS: dict[str, list[str]] = {
    "abstract": ["Abstract"],
    "title":    ["Title"],
}

# PAJAIS 25-category taxonomy (Pan-Pacific Journal of AIS)
PAJAIS_TAXONOMY: list[str] = [
    "Artificial Intelligence & Machine Learning",
    "Big Data & Analytics",
    "Blockchain & Distributed Ledger",
    "Cloud Computing & Infrastructure",
    "Cybersecurity & Privacy",
    "Decision Support Systems",
    "Digital Business & E-Commerce",
    "Digital Health & Telemedicine",
    "Digital Innovation & Transformation",
    "Enterprise Systems & ERP",
    "Fintech & Digital Finance",
    "Green IS & Sustainability",
    "Human-Computer Interaction",
    "Information Systems Strategy",
    "IT Governance & Management",
    "Knowledge Management",
    "Mobile Computing & IoT",
    "Natural Language Processing & Text Mining",
    "Organizational Behavior & IS",
    "Platform Ecosystems & APIs",
    "Privacy & Ethics in IS",
    "Smart Cities & Digital Government",
    "Social Media & Collaboration",
    "Supply Chain & Logistics IS",
    "Virtual Reality & Immersive Technologies",
]

# Boilerplate patterns to strip from abstracts
_BOILERPLATE_RE = re.compile(
    r"(©\s*\d{4}.*?(?:rights reserved|elsevier|springer|wiley)[^.]*\.?)"
    r"|(all rights reserved\.?)"
    r"|(published by.*?(?:ltd|inc|llc)[^.]*\.?)"
    r"|(doi:\s*\S+)",
    re.IGNORECASE,
)

# Sentence splitter — split on sentence-boundary punctuation, keep >= 20 chars
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# Private helpers  (pure functions, no side-effects)
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_dir(run_key: str) -> Path:
    return _ensure_dir(OUTPUT_DIR / run_key)


def _clean_text(text: str) -> str:
    return _BOILERPLATE_RE.sub("", str(text)).strip()


def _split_sentences(text: str) -> list[str]:
    return list(filter(
        lambda s: len(s.strip()) >= 20,
        _SENT_RE.split(_clean_text(text)),
    ))


def _embed(sentences: list[str]) -> np.ndarray:
    """Encode sentences to L2-normalised 384-d vectors."""
    model = SentenceTransformer(EMBED_MODEL)
    raw   = model.encode(sentences, show_progress_bar=False, batch_size=64)
    return normalize(raw, norm="l2")   # unit-norm -> cosine = dot product


def _cluster(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    return AgglomerativeClustering(
        metric="cosine",
        linkage="average",
        distance_threshold=threshold,
        n_clusters=None,
    ).fit_predict(embeddings)


def _centroid(embeddings: np.ndarray) -> np.ndarray:
    """Mean-pool rows then re-normalise to unit length."""
    return normalize(embeddings.mean(axis=0, keepdims=True), norm="l2")[0]


def _top_k_indices(embeddings: np.ndarray, centroid: np.ndarray, k: int) -> np.ndarray:
    sims = cosine_similarity(embeddings, centroid.reshape(1, -1)).flatten()
    return np.argsort(sims)[::-1][:k]


def _llm() -> ChatMistralAI:
    return ChatMistralAI(
        model=MODEL_NAME,
        api_key=MISTRAL_API_KEY,
        temperature=0.2,
        random_seed=RANDOM_SEED,
        timeout=LLM_TIMEOUT_S,
        max_retries=LLM_MAX_RETRIES,
    )


def _llm_groq():
    if ChatGroq is None:
        raise RuntimeError(
            "langchain-groq is not installed. Install dependencies from requirements.txt "
            "to enable Groq topic-label verification."
        )
    return ChatGroq(
        model=GROQ_MODEL_NAME,
        api_key=GROQ_API_KEY,
        temperature=0.2,
        timeout=LLM_TIMEOUT_S,
        max_retries=LLM_MAX_RETRIES,
    )


def _groq_enabled() -> bool:
    return bool(GROQ_API_KEY) and ChatGroq is not None


def _to_float(value: object, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _is_transient_provider_error(exc: Exception) -> bool:
    """Detect transient provider outages (Mistral/Groq) that should be retried."""
    msg = str(exc).lower()
    return (
        "unreachable_backend" in msg
        or "internal server error" in msg
        or '"code":"1100"' in msg
        or '"raw_status_code":503' in msg
        or '"raw_status_code":502' in msg
        or '"raw_status_code":504' in msg
        or '"status":503' in msg
        or '"status":502' in msg
        or '"status":504' in msg
        or '"status":429' in msg
        or "too many requests" in msg
        or "rate limit" in msg
        or "gateway timeout" in msg
        or "service unavailable" in msg
    )


def _invoke_with_retries(fn):
    """Run an LLM call with bounded linear backoff on transient provider errors."""
    last_exc: Exception | None = None
    for attempt in range(PROVIDER_RETRY_ATTEMPTS):
        try:
            return fn()
        except Exception as exc:
            if not _is_transient_provider_error(exc):
                raise
            last_exc = exc
            if attempt < PROVIDER_RETRY_ATTEMPTS - 1:
                time.sleep(PROVIDER_RETRY_BASE_DELAY_S * (attempt + 1))
                continue
            raise last_exc

    raise RuntimeError("Unexpected retry flow in _invoke_with_retries")


def _save_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Plotly chart builders
# ---------------------------------------------------------------------------

def _chart_intertopic(summaries: list[dict]) -> go.Figure:
    df = pd.DataFrame(summaries)
    return px.scatter(
        df,
        x="cx", y="cy",
        size="size",
        text="cluster_id",
        color="size",
        color_continuous_scale="Blues",
        title="Intertopic Distance Map",
        labels={"cx": "Dim-1", "cy": "Dim-2", "size": "Sentences"},
        template="plotly_dark",
    )


def _chart_top_words(summaries: list[dict]) -> go.Figure:
    df = (
        pd.DataFrame(summaries)
        .nlargest(20, "size")
        .assign(label=lambda d: d["cluster_id"].astype(str))
    )
    return px.bar(
        df,
        x="size", y="label",
        orientation="h",
        title="Top Clusters by Sentence Count",
        labels={"size": "Sentences", "label": "Cluster"},
        color="size",
        color_continuous_scale="Teal",
        template="plotly_dark",
    )


def _chart_hierarchy(labels: list[int], embeddings: np.ndarray) -> go.Figure:
    unique     = sorted(set(labels))
    labels_arr = np.array(labels)
    centroids  = np.vstack([
        _centroid(embeddings[labels_arr == lbl])
        for lbl in unique
    ])
    dist_mat = 1 - cosine_similarity(centroids)
    fig = ff.create_dendrogram(
        dist_mat,
        labels=[str(l) for l in unique],
        colorscale=px.colors.sequential.Blues,
    )
    fig.update_layout(title="Cluster Hierarchy", template="plotly_dark")
    return fig


def _chart_heatmap(labels: list[int], embeddings: np.ndarray) -> go.Figure:
    unique     = sorted(set(labels))
    labels_arr = np.array(labels)
    centroids  = np.vstack([
        _centroid(embeddings[labels_arr == lbl])
        for lbl in unique
    ])
    sim_mat = cosine_similarity(centroids)
    return px.imshow(
        sim_mat,
        x=[str(l) for l in unique],
        y=[str(l) for l in unique],
        color_continuous_scale="Blues",
        title="Cluster Similarity Heatmap",
        template="plotly_dark",
    )


def _save_chart(fig: go.Figure, path: Path) -> str:
    fig.write_html(str(path), full_html=True, include_plotlyjs="cdn")
    return str(path)


# ============================================================================
# TOOL 1 — load_scopus_csv
# ============================================================================

@tool
def load_scopus_csv(filepath: str) -> dict:
    """
    Load a Scopus-exported CSV and extract corpus statistics.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the CSV file.

    Returns
    -------
    dict with keys:
        paper_count, abstract_sentence_count, title_sentence_count,
        columns, sample_abstracts, filepath
    """
    df = pd.read_csv(filepath).rename(columns=str.strip)

    abstract_sentences = list(reduce(
        lambda acc, sents: acc + sents,
        map(_split_sentences, df["Abstract"].dropna().tolist()),
        [],
    ))

    title_sentences = list(reduce(
        lambda acc, sents: acc + sents,
        map(_split_sentences, df["Title"].dropna().tolist()),
        [],
    ))

    _ensure_dir(OUTPUT_DIR / "abstract")
    _ensure_dir(OUTPUT_DIR / "title")

    _save_json(OUTPUT_DIR / "abstract" / "sentences.json", abstract_sentences)
    _save_json(OUTPUT_DIR / "title"    / "sentences.json", title_sentences)

    df.to_csv(OUTPUT_DIR / "corpus.csv", index=False)

    return {
        "paper_count":             int(len(df)),
        "abstract_sentence_count": int(len(abstract_sentences)),
        "title_sentence_count":    int(len(title_sentences)),
        "columns":                 df.columns.tolist(),
        "sample_abstracts":        df["Abstract"].dropna().head(3).tolist(),
        "filepath":                str(filepath),
    }


# ============================================================================
# TOOL 2 — run_bertopic_discovery
# ============================================================================

@tool
def run_bertopic_discovery(run_key: str, threshold: float = DISTANCE_THRESH) -> dict:
    """
    Embed sentences, cluster with AgglomerativeClustering, extract evidence,
    and generate four Plotly charts.

    Saved artefacts
    ---------------
    emb.npy         : (N, 384) float32  L2-normalised embeddings
    sent_labels.npy : (N,)     int32    per-sentence cluster label  [BUG 1 FIX]
    summaries.json  : list of cluster dicts with evidence sentences

    Parameters
    ----------
    run_key   : str   — "abstract" or "title"
    threshold : float — cosine distance threshold for AgglomerativeClustering

    Returns
    -------
    dict with keys:
        run_key, n_clusters, n_sentences, threshold,
        chart_paths, summaries_path, embeddings_path
    """
    rdir      = _run_dir(run_key)
    sentences = _load_json(OUTPUT_DIR / run_key / "sentences.json")

    embeddings = _embed(sentences)
    np.save(str(rdir / "emb.npy"), embeddings)

    labels     = _cluster(embeddings, threshold).tolist()
    unique_ids = sorted(set(labels))

    # FIX BUG 1 — persist per-sentence label array so Tool 4 can build
    # correct cluster masks without any guesswork or scaffolding.
    np.save(str(rdir / "sent_labels.npy"), np.array(labels, dtype=np.int32))

    labels_arr = np.array(labels)

    def _cluster_summary(cid: int) -> dict:
        mask    = labels_arr == cid
        c_emb   = embeddings[mask]
        c_sent  = list(np.array(sentences)[mask])
        ctroid  = _centroid(c_emb)
        top_idx = _top_k_indices(c_emb, ctroid, N_EVIDENCE)
        return {
            "cluster_id": int(cid),
            "size":       int(mask.sum()),
            "cx":         float(ctroid[0]),
            "cy":         float(ctroid[1]),
            "evidence":   list(np.array(c_sent)[top_idx]),
        }

    summaries = list(map(_cluster_summary, unique_ids))
    _save_json(rdir / "summaries.json", summaries)

    chart_paths = {
        "Intertopic Map": _save_chart(_chart_intertopic(summaries),        rdir / "intertopic.html"),
        "Top Words":      _save_chart(_chart_top_words(summaries),          rdir / "topwords.html"),
        "Hierarchy":      _save_chart(_chart_hierarchy(labels, embeddings), rdir / "hierarchy.html"),
        "Heatmap":        _save_chart(_chart_heatmap(labels, embeddings),   rdir / "heatmap.html"),
    }

    return {
        "run_key":         run_key,
        "n_clusters":      int(len(unique_ids)),
        "n_sentences":     int(len(sentences)),
        "threshold":       threshold,
        "chart_paths":     chart_paths,
        "summaries_path":  str(rdir / "summaries.json"),
        "embeddings_path": str(rdir / "emb.npy"),
    }


# ============================================================================
# TOOL 3 — label_topics_with_llm
# ============================================================================

_LABEL_PROMPT = PromptTemplate.from_template(
    """You are an expert academic researcher specialising in Information Systems.

Given the following cluster of research sentences, return a JSON object with EXACTLY these keys:
  label      : short research-area name (<= 6 words)
  category   : broader IS research category
  confidence : float 0.0-1.0
  reasoning  : one sentence explaining your choice
  niche      : boolean - true if highly specialised / narrow

Cluster ID    : {cluster_id}
Sentence count: {size}
Evidence sentences:
{evidence}

Respond with RAW JSON only. No markdown, no explanation outside the JSON.
"""
)


@tool
def label_topics_with_llm(run_key: str) -> dict:
    """
    Label each cluster with Mistral only (default Phase 2 labeling pass).

    Parameters
    ----------
    run_key : str — "abstract" or "title"

    Returns
    -------
    dict with keys:
        run_key, labels_path, labelled_count, labels_preview (list of dicts)
    """
    rdir      = _run_dir(run_key)
    summaries_path = rdir / "summaries.json"
    if not summaries_path.exists():
        return {
            "run_key":           run_key,
            "labels_path":       str(rdir / "labels.json"),
            "labelled_count":    0,
            "total_clusters":    0,
            "selected_clusters": 0,
            "skipped_clusters":  0,
            "labels_preview":    [],
            "error": (
                f"Missing discovery artifact: {summaries_path}. "
                "Run run_bertopic_discovery first for this run_key."
            ),
        }

    summaries = _load_json(summaries_path)

    ranked = sorted(
        filter(lambda s: s.get("size", 0) >= MIN_CLUSTER_SIZE_FOR_LABEL, summaries),
        key=lambda s: s.get("size", 0),
        reverse=True,
    )
    selected = ranked[:MAX_LABEL_CLUSTERS]

    chain_mistral = _LABEL_PROMPT | _llm() | JsonOutputParser()

    def _evidence_block(summary: dict) -> str:
        return "\n".join(
            f"  {i+1}. {s}"
            for i, s in enumerate(summary["evidence"])
        )

    def _label_one(summary: dict) -> dict:
        result = _invoke_with_retries(lambda: chain_mistral.invoke({
            "cluster_id": summary["cluster_id"],
            "size":       summary["size"],
            "evidence":   _evidence_block(summary),
        }))

        return {
            **summary,
            **result,
            "mistral_label":      result.get("label", ""),
            "mistral_category":   result.get("category", ""),
            "mistral_confidence": _to_float(result.get("confidence"), 0.0),
            "mistral_reasoning":  result.get("reasoning", ""),
            "mistral_niche":      bool(result.get("niche", False)),
            "groq_label":         "",
            "groq_category":      "",
            "groq_confidence":    0.0,
            "groq_reasoning":     "",
            "groq_niche":         False,
            "verification_done":  False,
            "verification_note":  "Run VERIFY in Phase 2 to compare with Groq labels.",
        }

    labelled = list(map(_label_one, selected))
    _save_json(rdir / "labels.json", labelled)

    # Keep tool output compact so the ReAct transcript does not overflow model context.
    preview = list(map(
        lambda r: {
            "cluster_id": r.get("cluster_id"),
            "label":         r.get("label"),
            "category":      r.get("category"),
            "confidence":    r.get("confidence"),
            "mistral_label": r.get("mistral_label", ""),
            "groq_label":    r.get("groq_label", ""),
            "size":          r.get("size"),
            "niche":         r.get("niche", False),
        },
        labelled[:MAX_TOOL_RETURN_PREVIEW],
    ))

    return {
        "run_key":           run_key,
        "labels_path":       str(rdir / "labels.json"),
        "labelled_count":    len(labelled),
        "total_clusters":    len(summaries),
        "selected_clusters": len(selected),
        "skipped_clusters":  max(0, len(summaries) - len(selected)),
        "groq_enabled":      _groq_enabled(),
        "mode_note":         "Single-model labeling complete (Mistral). Send VERIFY in Phase 2 to run Groq verification.",
        "labels_preview":    preview,
    }


@tool
def verify_topic_labels_with_groq(run_key: str) -> dict:
    """
    Run Groq topic labeling for already-labeled topics and append comparison fields
    into labels.json so UI review table can show both Mistral and Groq labels.

    Parameters
    ----------
    run_key : str — "abstract" or "title"

    Returns
    -------
    dict with keys:
        run_key, labels_path, verification_path, verified_count, labels_preview
    """
    rdir          = _run_dir(run_key)
    labels_path   = rdir / "labels.json"
    summaries_path = rdir / "summaries.json"

    if not _groq_enabled():
        return {
            "run_key": run_key,
            "labels_path": str(labels_path),
            "verified_count": 0,
            "labels_preview": [],
            "error": (
                "GROQ_API_KEY is missing or langchain-groq is unavailable. "
                "Set GROQ_API_KEY and install requirements to use VERIFY."
            ),
        }

    if not labels_path.exists():
        return {
            "run_key": run_key,
            "labels_path": str(labels_path),
            "verified_count": 0,
            "labels_preview": [],
            "error": (
                f"Missing labels artifact: {labels_path}. "
                "Run label_topics_with_llm first."
            ),
        }

    if not summaries_path.exists():
        return {
            "run_key": run_key,
            "labels_path": str(labels_path),
            "verified_count": 0,
            "labels_preview": [],
            "error": (
                f"Missing summaries artifact: {summaries_path}. "
                "Run run_bertopic_discovery first."
            ),
        }

    labels_data = _load_json(labels_path)
    summaries = _load_json(summaries_path)
    summary_by_id = {
        int(s.get("cluster_id", -1)): s
        for s in summaries
    }

    target_rows = list(filter(
        lambda r: int(r.get("cluster_id", -1)) in summary_by_id,
        labels_data,
    ))

    chain_groq = _LABEL_PROMPT | _llm_groq() | JsonOutputParser()

    def _evidence_block(summary: dict) -> str:
        return "\n".join(
            f"  {i+1}. {s}"
            for i, s in enumerate(summary.get("evidence", []))
        )

    def _label_with_groq(row: dict) -> tuple[int, dict]:
        cid = int(row.get("cluster_id", -1))
        summary = summary_by_id[cid]
        result = _invoke_with_retries(lambda: chain_groq.invoke({
            "cluster_id": summary["cluster_id"],
            "size":       summary["size"],
            "evidence":   _evidence_block(summary),
        }))
        return cid, result

    groq_pairs = list(map(_label_with_groq, target_rows))
    groq_by_id = {cid: data for cid, data in groq_pairs}

    def _merge_row(row: dict) -> dict:
        cid = int(row.get("cluster_id", -1))
        groq = groq_by_id.get(cid, {})
        has_groq = bool(groq)
        mistral_label = str(row.get("mistral_label") or row.get("label", "")).strip()
        groq_label = str(groq.get("label", "")).strip()
        is_agreement = (
            mistral_label.lower() == groq_label.lower()
            if has_groq and mistral_label and groq_label
            else False
        )

        return {
            **row,
            "mistral_label":      mistral_label,
            "mistral_category":   row.get("mistral_category") or row.get("category", ""),
            "mistral_confidence": _to_float(
                row.get("mistral_confidence", row.get("confidence", 0.0)),
                0.0,
            ),
            "mistral_reasoning":  row.get("mistral_reasoning") or row.get("reasoning", ""),
            "mistral_niche":      bool(row.get("mistral_niche", row.get("niche", False))),
            "groq_label":         groq.get("label", ""),
            "groq_category":      groq.get("category", ""),
            "groq_confidence":    _to_float(groq.get("confidence"), 0.0),
            "groq_reasoning":     groq.get("reasoning", ""),
            "groq_niche":         bool(groq.get("niche", False)),
            "verification_done":  has_groq,
            "verification_note": (
                "Mistral and Groq labels match."
                if is_agreement
                else "Mistral and Groq labels differ. Review before approval."
            )
            if has_groq
            else "Groq labeling unavailable for this topic.",
        }

    verified_rows = list(map(_merge_row, labels_data))
    verification_path = rdir / "labels_verification.json"
    _save_json(labels_path, verified_rows)
    _save_json(verification_path, verified_rows)

    preview = list(map(
        lambda r: {
            "cluster_id":    r.get("cluster_id"),
            "mistral_label": r.get("mistral_label", ""),
            "groq_label":    r.get("groq_label", ""),
            "verification_note": r.get("verification_note", ""),
        },
        verified_rows[:MAX_TOOL_RETURN_PREVIEW],
    ))

    verified_count = sum(1 for row in verified_rows if row.get("groq_label"))

    return {
        "run_key":           run_key,
        "labels_path":       str(labels_path),
        "verification_path": str(verification_path),
        "verified_count":    int(verified_count),
        "labelled_count":    int(len(verified_rows)),
        "labels_preview":    preview,
    }


# ============================================================================
# TOOL 4 — consolidate_into_themes
# ============================================================================

@tool
def consolidate_into_themes(run_key: str, theme_map: dict) -> dict:
    """
    Merge approved / renamed topics into consolidated themes and recompute
    centroids from the actual merged-cluster embeddings.

    Parameters
    ----------
    run_key   : str  — "abstract" or "title"
    theme_map : dict — {new_theme_name: [cluster_id, ...], ...}
                       Only approved topics need appear here.

    Returns
    -------
    dict with keys:
        run_key, theme_count, themes_path, themes_preview (list of dicts)
    """
    rdir        = _run_dir(run_key)
    labels_data = _load_json(rdir / "labels.json")
    embeddings  = np.load(str(rdir / "emb.npy"))          # (N, 384)
    sent_labels = np.load(str(rdir / "sent_labels.npy"))  # (N,) — FIX BUG 1

    # Index label dicts by cluster_id for O(1) lookup
    label_idx = {item["cluster_id"]: item for item in labels_data}

    def _build_theme(theme_name: str, cids: list[int]) -> dict:
        """
        Build one consolidated theme from a list of cluster IDs.

        Evidence : top-N sentences pooled across all merged clusters
        Centroid : L2-normalised mean of all embeddings in the merged set
        Size     : total sentence count across merged clusters
        """
        member_labels = list(map(label_idx.get, cids))

        # Pool evidence sentences from all member clusters
        all_evidence = reduce(
            lambda acc, lbl: acc + lbl["evidence"],
            filter(None, member_labels),
            [],
        )

        # Total sentence count across merged clusters
        total_size = reduce(
            lambda acc, lbl: acc + lbl.get("size", 0),
            filter(None, member_labels),
            0,
        )

        # FIX BUG 1 — build correct cluster mask using persisted sent_labels
        cluster_mask     = np.isin(sent_labels, np.array(cids, dtype=np.int32))
        theme_embeddings = embeddings[cluster_mask]   # (M, 384)

        # Guard: if mask is somehow empty fall back to zero vector
        theme_centroid = (
            _centroid(theme_embeddings)
            if theme_embeddings.shape[0] > 0
            else np.zeros(embeddings.shape[1], dtype=np.float32)
        )

        return {
            "theme_name":  theme_name,
            "cluster_ids": cids,
            "size":        total_size,
            "evidence":    all_evidence[:N_EVIDENCE],
            "centroid":    theme_centroid.tolist(),
            "sub_labels":  list(map(
                               itemgetter("label"),
                               filter(None, member_labels),
                           )),
        }

    themes = list(map(
        lambda kv: _build_theme(kv[0], kv[1]),
        theme_map.items(),
    ))

    _save_json(rdir / "themes.json", themes)

    preview = list(map(
        lambda t: {
            "theme_name":   t.get("theme_name"),
            "size":         t.get("size", 0),
            "cluster_count": len(t.get("cluster_ids", [])),
        },
        themes[:MAX_TOOL_RETURN_PREVIEW],
    ))

    return {
        "run_key":     run_key,
        "theme_count": len(themes),
        "themes_path": str(rdir / "themes.json"),
        "themes_preview": preview,
    }


# ============================================================================
# TOOL 5 — compare_with_taxonomy
# ============================================================================

_TAXONOMY_PROMPT = PromptTemplate.from_template(
    """You are an IS research taxonomist. Map the following research theme to the
PAJAIS taxonomy. Return RAW JSON with EXACTLY these keys:
  theme_name    : the input theme name (unchanged)
  pajais_match  : best matching PAJAIS category OR the string "NOVEL"
  confidence    : float 0.0-1.0
  reasoning     : one sentence
  is_novel      : boolean

PAJAIS categories:
{taxonomy}

Theme to map:
  Name     : {theme_name}
  Evidence : {evidence}

Respond with RAW JSON only. No markdown.
"""
)


@tool
def compare_with_taxonomy(run_key: str) -> dict:
    """
    Map consolidated themes to PAJAIS taxonomy via Mistral.

    Parameters
    ----------
    run_key : str — "abstract" or "title"

    Returns
    -------
    dict with keys:
        run_key, taxonomy_path, mapped_count, novel_count, mapping_preview
    """
    rdir   = _run_dir(run_key)
    themes = _load_json(rdir / "themes.json")
    chain  = _TAXONOMY_PROMPT | _llm() | JsonOutputParser()

    taxonomy_str = "\n".join(f"  - {cat}" for cat in PAJAIS_TAXONOMY)

    def _map_theme(theme: dict) -> dict:
        result = _invoke_with_retries(lambda: chain.invoke({
            "taxonomy":   taxonomy_str,
            "theme_name": theme["theme_name"],
            "evidence":   " | ".join(theme.get("evidence", [])[:3]),
        }))
        return {**theme, **result}

    taxonomy_map = list(map(_map_theme, themes))
    _save_json(rdir / "taxonomy_map.json", taxonomy_map)

    novel_count  = sum(1 for t in taxonomy_map if t.get("is_novel", False))
    mapped_count = len(taxonomy_map) - novel_count

    preview = list(map(
        lambda t: {
            "theme_name":   t.get("theme_name"),
            "pajais_match": t.get("pajais_match", "NOVEL"),
            "confidence":   t.get("confidence", 0),
            "is_novel":     t.get("is_novel", False),
        },
        taxonomy_map[:MAX_TOOL_RETURN_PREVIEW],
    ))

    return {
        "run_key":       run_key,
        "taxonomy_path": str(rdir / "taxonomy_map.json"),
        "mapped_count":  mapped_count,
        "novel_count":   novel_count,
        "mapping_preview": preview,
    }


# ============================================================================
# TOOL 6 — generate_comparison_csv
# ============================================================================

@tool
def generate_comparison_csv() -> dict:
    """
    Side-by-side comparison of abstract-run vs title-run themes.

    FIX ISSUE 1: title run is optional — no longer crashes if only the
    abstract run has been completed. title_map defaults to [] when the
    title taxonomy_map.json file does not exist.

    Returns
    -------
    dict with keys:
        csv_path, row_count, columns, preview (list of dicts)
    """
    abstract_path = OUTPUT_DIR / "abstract" / "taxonomy_map.json"
    title_path    = OUTPUT_DIR / "title"    / "taxonomy_map.json"

    abstract_map = _load_json(abstract_path)

    # FIX ISSUE 1: guard against missing title run
    title_map = (
        _load_json(title_path)
        if title_path.exists()
        else []
    )

    def _row(a_theme: dict, t_theme: dict | None) -> dict:
        return {
            "Abstract Theme":      a_theme.get("theme_name",   ""),
            "Abstract PAJAIS":     a_theme.get("pajais_match",  ""),
            "Abstract Confidence": a_theme.get("confidence",    0),
            "Abstract Novel":      a_theme.get("is_novel",     False),
            "Title Theme":         t_theme.get("theme_name",   "") if t_theme else "",
            "Title PAJAIS":        t_theme.get("pajais_match",  "") if t_theme else "",
            "Title Confidence":    t_theme.get("confidence",    0)  if t_theme else 0,
            "Title Novel":         t_theme.get("is_novel",     False) if t_theme else False,
        }

    max_len  = max(len(abstract_map), len(title_map)) if title_map else len(abstract_map)
    padded_a = abstract_map + [{}] * (max_len - len(abstract_map))
    padded_t = title_map    + [{}] * (max_len - len(title_map))

    rows = list(map(_row, padded_a, padded_t))
    df   = pd.DataFrame(rows)

    out_path = OUTPUT_DIR / "comparison.csv"
    df.to_csv(out_path, index=False)

    return {
        "csv_path":  str(out_path),
        "row_count": len(df),
        "columns":   df.columns.tolist(),
        "preview":   df.head(5).to_dict(orient="records"),
    }


# ============================================================================
# TOOL 7 — export_narrative
# ============================================================================

_NARRATIVE_PROMPT = PromptTemplate.from_template(
    """You are an academic researcher writing a methodology and findings section.

Write a 500-word academic narrative describing the thematic analysis results below.
Structure: (1) methodology overview, (2) major themes found, (3) PAJAIS alignment,
(4) novel contributions, (5) limitations.

Use formal academic English. Do NOT use bullet points.

Abstract themes & taxonomy:
{abstract_themes}

Title themes & taxonomy:
{title_themes}

Respond with plain text only.
"""
)


@tool
def export_narrative(run_key: str) -> dict:
    """
    Generate a 500-word academic narrative and save to narrative.txt.

    Parameters
    ----------
    run_key : str — "abstract" or "title" (primary source)

    Returns
    -------
    dict with keys:
        narrative_path, word_count, preview (first 300 chars)
    """
    rdir       = _run_dir(run_key)
    title_path = OUTPUT_DIR / "title" / "taxonomy_map.json"

    abstract_map = _load_json(OUTPUT_DIR / "abstract" / "taxonomy_map.json")
    title_map    = _load_json(title_path) if title_path.exists() else []

    def _theme_summary(t: dict) -> str:
        return (
            f"  - {t.get('theme_name','?')} -> {t.get('pajais_match','?')} "
            f"(conf={t.get('confidence',0):.2f}, novel={t.get('is_novel',False)})"
        )

    abstract_str = "\n".join(map(_theme_summary, abstract_map))
    title_str    = "\n".join(map(_theme_summary, title_map)) or "Not run."

    chain    = _NARRATIVE_PROMPT | _llm()
    response = _invoke_with_retries(lambda: chain.invoke({
        "abstract_themes": abstract_str,
        "title_themes":    title_str,
    }))

    narrative = response.content if hasattr(response, "content") else str(response)
    out_path  = rdir / "narrative.txt"
    out_path.write_text(narrative, encoding="utf-8")

    return {
        "narrative_path": str(out_path),
        "word_count":     len(narrative.split()),
        "preview":        narrative[:300],
    }


# ---------------------------------------------------------------------------
# Tool registry — imported by agent.py
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    load_scopus_csv,
    run_bertopic_discovery,
    label_topics_with_llm,
    verify_topic_labels_with_groq,
    consolidate_into_themes,
    compare_with_taxonomy,
    generate_comparison_csv,
    export_narrative,
]
