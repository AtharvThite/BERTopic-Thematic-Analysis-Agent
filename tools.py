"""
tools.py — BERTopic Thematic Analysis Pipeline Tools
=====================================================
Nine LangChain @tool functions implementing Braun & Clarke's (2006)
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
    pip install langchain langchain-core langchain-mistralai langchain-groq
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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
import hdbscan
import umap

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
GROQ_OLLAMA_MODEL_NAME: str = os.environ.get("GROQ_OLLAMA_MODEL_NAME", "llama-3.3-70b-versatile")
GROQ_GPT_MODEL_NAME: str    = os.environ.get("GROQ_GPT_MODEL_NAME", "openai/gpt-oss-120b")
GROQ_JUDGE_MODEL_NAME: str  = os.environ.get("GROQ_JUDGE_MODEL_NAME", "llama-3.1-8b-instant")
EMBED_MODEL:     str   = "allenai/specter2_base"
BASE_DIR:        Path  = Path(__file__).resolve().parent
OUTPUT_DIR:      Path  = BASE_DIR / "outputs"
N_EVIDENCE:      int   = 5       # sentences kept per cluster centroid
DISTANCE_THRESH: float = 0.35   # cosine-distance threshold (1 - similarity)
RANDOM_SEED:     int   = 42
LLM_TIMEOUT_S:   int   = 45
LLM_MAX_RETRIES: int   = 3
MAX_LABEL_CLUSTERS: int = 60
MIN_CLUSTER_SIZE_FOR_LABEL: int = 20
MAX_TOOL_RETURN_PREVIEW: int = 12
PROVIDER_RETRY_ATTEMPTS: int = 4
PROVIDER_RETRY_BASE_DELAY_S: float = 2.0
PROVIDER_RETRY_RATE_LIMIT_DELAY_S: float = 6.0
PROVIDER_RETRY_MAX_DELAY_S: float = 18.0
HDBSCAN_MIN_CLUSTER_SIZE: int = 20
HDBSCAN_MIN_SAMPLES: int = 5
HDBSCAN_MAX_CLUSTER_SIZE: int = 120
UMAP_N_NEIGHBORS: int = 15
UMAP_MIN_DIST: float = 0.0
UMAP_N_COMPONENTS_CLUSTER: int = 5
UMAP_N_COMPONENTS_VIZ: int = 2
AUTO_OPTIMIZE_CLUSTERS: bool = True
OPTIMIZE_MAX_ITERS: int = 6
OPTIMIZE_STABLE_ROUNDS: int = 2
OPTIMIZE_MIN_IMPROVEMENT: float = 0.01
OPTIMIZE_TARGET_CLUSTER_MIN: int = 20
OPTIMIZE_TARGET_CLUSTER_MAX: int = 120
OPTIMIZE_TARGET_NOISE_MAX: float = 0.50
OPTIMIZE_MIN_CLUSTER_SIZE_MIN: int = 5
OPTIMIZE_MIN_CLUSTER_SIZE_MAX: int = 60
OPTIMIZE_MAX_CLUSTER_SIZE_MIN: int = 40
OPTIMIZE_MAX_CLUSTER_SIZE_MAX: int = 200
OPTIMIZE_MIN_SAMPLES_MIN: int = 1
OPTIMIZE_MIN_SAMPLES_MAX: int = 15

# Run configurations — keys map to source columns
RUN_CONFIGS: dict[str, list[str]] = {
    "abstract": ["Abstract"],
    "title":    ["Title"],
    "keywords": [
        "Author Keywords",
        "Author Keywords Plus",
        "Index Keywords",
        "Keywords",
        "Author_Keywords",
    ],
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
_KEYWORD_SPLIT_RE = re.compile(r"\s*[;|]\s*")
_KEYWORD_COMMA_RE = re.compile(r"\s*,\s*")


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


def _split_keywords(text: str) -> list[str]:
    cleaned = _clean_text(text).replace("\n", " ").strip()
    if not cleaned:
        return []
    primary = list(filter(None, map(str.strip, _KEYWORD_SPLIT_RE.split(cleaned))))
    terms = (
        primary
        if len(primary) > 1
        else list(filter(None, map(str.strip, _KEYWORD_COMMA_RE.split(cleaned))))
    )
    return list(dict.fromkeys(filter(lambda t: len(t) >= 2, terms)))


def _resolve_column_name(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalised = {
        str(col).strip().lower(): col
        for col in df.columns
    }
    return next(
        (normalised.get(str(c).strip().lower()) for c in candidates
         if normalised.get(str(c).strip().lower()) is not None),
        None,
    )


def _texts_for_candidates(df: pd.DataFrame, candidates: list[str]) -> tuple[list[str], str | None]:
    col = _resolve_column_name(df, candidates)
    return (
        df[col].dropna().astype(str).tolist(),
        col,
    ) if col else ([], None)


def _embed(sentences: list[str]) -> np.ndarray:
    """Encode sentences to L2-normalised SPECTER2 vectors."""
    model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
    raw   = model.encode(sentences, show_progress_bar=False, batch_size=64)
    return normalize(raw, norm="l2")   # unit-norm -> cosine = dot product


def _umap_reduce(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=n_components,
        metric="cosine",
        random_state=RANDOM_SEED,
    )
    return reducer.fit_transform(embeddings)


def _cluster(embeddings: np.ndarray,
             min_cluster_size: int,
             max_cluster_size: int,
             min_samples: int) -> np.ndarray:
    return hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        max_cluster_size=max_cluster_size,
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


def _llm_groq(model_name: str):
    if ChatGroq is None:
        raise RuntimeError(
            "langchain-groq is not installed. Install dependencies from requirements.txt "
            "to enable Groq topic-label verification."
        )
    return ChatGroq(
        model=model_name,
        api_key=GROQ_API_KEY,
        temperature=0.2,
        timeout=LLM_TIMEOUT_S,
        max_retries=LLM_MAX_RETRIES,
    )


def _groq_ollama_enabled() -> bool:
    return bool(GROQ_API_KEY) and ChatGroq is not None and bool(GROQ_OLLAMA_MODEL_NAME)


def _groq_gpt_enabled() -> bool:
    return bool(GROQ_API_KEY) and ChatGroq is not None and bool(GROQ_GPT_MODEL_NAME)


def _groq_judge_enabled() -> bool:
    return bool(GROQ_API_KEY) and ChatGroq is not None and bool(GROQ_JUDGE_MODEL_NAME)


def _to_float(value: object, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _clamp_int(value: object, low: int, high: int, fallback: int) -> int:
    try:
        casted = int(value)
    except (TypeError, ValueError):
        casted = int(fallback)
    return max(low, min(high, casted))


def _cluster_metrics(labels: np.ndarray) -> dict:
    labels_arr = np.array(labels, dtype=np.int32)
    n_sentences = int(labels_arr.shape[0])
    noise_count = int((labels_arr == -1).sum())
    unique_ids = sorted(filter(lambda v: v != -1, set(labels_arr.tolist())))
    sizes = list(map(lambda cid: int((labels_arr == cid).sum()), unique_ids))

    if sizes:
        min_size = float(np.min(sizes))
        median_size = float(np.median(sizes))
        mean_size = float(np.mean(sizes))
        max_size = float(np.max(sizes))
    else:
        min_size = 0.0
        median_size = 0.0
        mean_size = 0.0
        max_size = 0.0

    return {
        "n_sentences": n_sentences,
        "n_clusters": int(len(unique_ids)),
        "noise_ratio": float(noise_count) / float(max(1, n_sentences)),
        "min_size": min_size,
        "median_size": median_size,
        "mean_size": mean_size,
        "max_size": max_size,
    }


def _heuristic_hdbscan_tweak(metrics: dict, params: dict) -> dict:
    n_clusters = int(metrics.get("n_clusters", 0))
    noise_ratio = float(metrics.get("noise_ratio", 0.0))

    min_cluster_size = int(params.get("min_cluster_size", HDBSCAN_MIN_CLUSTER_SIZE))
    max_cluster_size = int(params.get("max_cluster_size", HDBSCAN_MAX_CLUSTER_SIZE))
    min_samples = int(params.get("min_samples", HDBSCAN_MIN_SAMPLES))

    action = "accept"
    reasoning = "Cluster metrics are within target ranges."

    if n_clusters < OPTIMIZE_TARGET_CLUSTER_MIN:
        min_cluster_size = max(
            OPTIMIZE_MIN_CLUSTER_SIZE_MIN,
            int(round(min_cluster_size * 0.8)),
        )
        min_samples = max(OPTIMIZE_MIN_SAMPLES_MIN, min_samples - 1)
        action = "tweak"
        reasoning = "Too few clusters; reducing min_cluster_size and min_samples."
    elif n_clusters > OPTIMIZE_TARGET_CLUSTER_MAX:
        min_cluster_size = min(
            OPTIMIZE_MIN_CLUSTER_SIZE_MAX,
            int(round(min_cluster_size * 1.2)),
        )
        min_samples = min(OPTIMIZE_MIN_SAMPLES_MAX, min_samples + 1)
        action = "tweak"
        reasoning = "Too many clusters; increasing min_cluster_size and min_samples."
    elif noise_ratio > OPTIMIZE_TARGET_NOISE_MAX:
        min_cluster_size = max(
            OPTIMIZE_MIN_CLUSTER_SIZE_MIN,
            int(round(min_cluster_size * 0.85)),
        )
        min_samples = max(OPTIMIZE_MIN_SAMPLES_MIN, min_samples - 1)
        action = "tweak"
        reasoning = "Noise ratio is high; lowering min_cluster_size and min_samples."

    return {
        "action": action,
        "min_cluster_size": min_cluster_size,
        "max_cluster_size": max_cluster_size,
        "min_samples": min_samples,
        "reasoning": reasoning,
    }


def _normalize_hdbscan_suggestion(suggestion: dict, current: dict) -> dict:
    action = str(suggestion.get("action", "accept")).strip().lower()
    action = action if action in {"accept", "tweak"} else "accept"

    min_cluster_size = _clamp_int(
        suggestion.get("min_cluster_size", current.get("min_cluster_size")),
        OPTIMIZE_MIN_CLUSTER_SIZE_MIN,
        OPTIMIZE_MIN_CLUSTER_SIZE_MAX,
        current.get("min_cluster_size", HDBSCAN_MIN_CLUSTER_SIZE),
    )
    max_cluster_size = _clamp_int(
        suggestion.get("max_cluster_size", current.get("max_cluster_size")),
        OPTIMIZE_MAX_CLUSTER_SIZE_MIN,
        OPTIMIZE_MAX_CLUSTER_SIZE_MAX,
        current.get("max_cluster_size", HDBSCAN_MAX_CLUSTER_SIZE),
    )
    min_samples = _clamp_int(
        suggestion.get("min_samples", current.get("min_samples")),
        OPTIMIZE_MIN_SAMPLES_MIN,
        OPTIMIZE_MIN_SAMPLES_MAX,
        current.get("min_samples", HDBSCAN_MIN_SAMPLES),
    )

    if max_cluster_size < min_cluster_size:
        max_cluster_size = min_cluster_size + 1

    return {
        "action": action,
        "min_cluster_size": min_cluster_size,
        "max_cluster_size": max_cluster_size,
        "min_samples": min_samples,
        "reasoning": str(suggestion.get("reasoning", "")).strip(),
    }


def _metrics_in_target(metrics: dict) -> bool:
    n_clusters = int(metrics.get("n_clusters", 0))
    noise_ratio = float(metrics.get("noise_ratio", 1.0))
    return (
        OPTIMIZE_TARGET_CLUSTER_MIN <= n_clusters <= OPTIMIZE_TARGET_CLUSTER_MAX
        and noise_ratio <= OPTIMIZE_TARGET_NOISE_MAX
    )


def _optimization_score(metrics: dict) -> float:
    n_clusters = int(metrics.get("n_clusters", 0))
    noise_ratio = float(metrics.get("noise_ratio", 1.0))

    if n_clusters < OPTIMIZE_TARGET_CLUSTER_MIN:
        cluster_penalty = (OPTIMIZE_TARGET_CLUSTER_MIN - n_clusters) / max(
            OPTIMIZE_TARGET_CLUSTER_MIN,
            1,
        )
    elif n_clusters > OPTIMIZE_TARGET_CLUSTER_MAX:
        cluster_penalty = (n_clusters - OPTIMIZE_TARGET_CLUSTER_MAX) / max(
            OPTIMIZE_TARGET_CLUSTER_MAX,
            1,
        )
    else:
        cluster_penalty = 0.0

    noise_penalty = max(0.0, noise_ratio - OPTIMIZE_TARGET_NOISE_MAX) / max(
        OPTIMIZE_TARGET_NOISE_MAX,
        1e-6,
    )

    return 1.0 - min(1.0, cluster_penalty + noise_penalty)


def _load_sentence_meta(run_key: str, sentences: list[str]) -> list[dict]:
    meta_path = OUTPUT_DIR / run_key / "sentence_meta.json"
    if not meta_path.exists():
        return [
            {
                "sentence": s,
                "paper_title": "",
                "paper_id": None,
            }
            for s in sentences
        ]

    meta = _load_json(meta_path)
    if not isinstance(meta, list):
        return [
            {
                "sentence": s,
                "paper_title": "",
                "paper_id": None,
            }
            for s in sentences
        ]

    if len(meta) != len(sentences):
        return [
            {
                "sentence": s,
                "paper_title": "",
                "paper_id": None,
            }
            for s in sentences
        ]

    return meta


def _top_papers_for_mask(meta: list[dict], mask: np.ndarray, k: int = 3) -> dict:
    counts: dict[tuple[object, str], int] = {}
    for idx, entry in enumerate(meta):
        if not mask[idx]:
            continue
        paper_id = entry.get("paper_id")
        title = str(entry.get("paper_title") or entry.get("title") or "").strip()
        if not title:
            title = f"Paper {paper_id}" if paper_id is not None else "Unknown"
        key = (paper_id, title)
        counts[key] = counts.get(key, 0) + 1

    ordered = sorted(
        counts.items(),
        key=lambda kv: (-kv[1], str(kv[0][1]).lower()),
    )

    top = [
        {"paper_id": pid, "paper_title": title, "count": count}
        for (pid, title), count in ordered[:k]
    ]

    return {
        "paper_count": int(len(counts)),
        "top_papers": top,
    }


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


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "rate limit" in msg
        or "too many requests" in msg
        or '"raw_status_code":429' in msg
        or '"status":429' in msg
        or "status code: 429" in msg
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
                delay = PROVIDER_RETRY_BASE_DELAY_S * (attempt + 1)
                if _is_rate_limit_error(exc):
                    delay = max(delay, PROVIDER_RETRY_RATE_LIMIT_DELAY_S * (attempt + 1))
                time.sleep(min(PROVIDER_RETRY_MAX_DELAY_S, delay))
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
    unique     = sorted(filter(lambda v: v != -1, set(labels)))
    if not unique:
        fig = go.Figure()
        fig.update_layout(title="Cluster Hierarchy", template="plotly_dark")
        return fig
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
    unique     = sorted(filter(lambda v: v != -1, set(labels)))
    if not unique:
        fig = go.Figure()
        fig.update_layout(title="Cluster Similarity Heatmap", template="plotly_dark")
        return fig
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


_OPTIMIZE_PROMPT = PromptTemplate.from_template(
    """You are optimizing HDBSCAN clustering parameters for BERTopic.

Current parameters:
  min_cluster_size: {min_cluster_size}
  max_cluster_size: {max_cluster_size}
  min_samples: {min_samples}

Clustering metrics:
  n_sentences: {n_sentences}
  n_clusters: {n_clusters}
  noise_ratio: {noise_ratio}
  min_size: {min_size}
  median_size: {median_size}
  mean_size: {mean_size}
  max_size: {max_size}

Constraints:
- Only adjust min_cluster_size, max_cluster_size, min_samples.
- Keep min_cluster_size within [{min_cluster_size_min}, {min_cluster_size_max}].
- Keep max_cluster_size within [{max_cluster_size_min}, {max_cluster_size_max}].
- Keep min_samples within [{min_samples_min}, {min_samples_max}].
- Prefer n_clusters in [{target_cluster_min}, {target_cluster_max}].
- Prefer noise_ratio <= {target_noise_max}.

Return RAW JSON with exactly these keys:
  action: "accept" or "tweak"
  min_cluster_size: int
  max_cluster_size: int
  min_samples: int
  reasoning: short sentence

If clustering already looks good, set action="accept" and repeat the current values.
Respond with RAW JSON only.
"""
)


def _recommend_hdbscan_params(metrics: dict, params: dict) -> dict:
    if not MISTRAL_API_KEY:
        return _normalize_hdbscan_suggestion(
            _heuristic_hdbscan_tweak(metrics, params),
            params,
        )

    chain = _OPTIMIZE_PROMPT | _llm() | JsonOutputParser()

    payload = {
        **metrics,
        **params,
        "min_cluster_size_min": OPTIMIZE_MIN_CLUSTER_SIZE_MIN,
        "min_cluster_size_max": OPTIMIZE_MIN_CLUSTER_SIZE_MAX,
        "max_cluster_size_min": OPTIMIZE_MAX_CLUSTER_SIZE_MIN,
        "max_cluster_size_max": OPTIMIZE_MAX_CLUSTER_SIZE_MAX,
        "min_samples_min": OPTIMIZE_MIN_SAMPLES_MIN,
        "min_samples_max": OPTIMIZE_MIN_SAMPLES_MAX,
        "target_cluster_min": OPTIMIZE_TARGET_CLUSTER_MIN,
        "target_cluster_max": OPTIMIZE_TARGET_CLUSTER_MAX,
        "target_noise_max": OPTIMIZE_TARGET_NOISE_MAX,
    }

    try:
        suggestion = _invoke_with_retries(lambda: chain.invoke(payload))
    except Exception:
        suggestion = {}

    if not isinstance(suggestion, dict) or not suggestion:
        suggestion = _heuristic_hdbscan_tweak(metrics, params)

    return _normalize_hdbscan_suggestion(suggestion, params)


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
        keywords_term_count,
        columns, sample_abstracts, filepath
    """
    df = pd.read_csv(filepath).rename(columns=str.strip)

    abstract_texts, abstract_col = _texts_for_candidates(df, RUN_CONFIGS["abstract"])
    title_texts, title_col       = _texts_for_candidates(df, RUN_CONFIGS["title"])
    keywords_texts, keywords_col = _texts_for_candidates(df, RUN_CONFIGS["keywords"])

    titles_for_meta = (
        df[title_col].fillna("").astype(str).tolist()
        if title_col
        else [""] * len(df)
    )

    def _build_sentences_and_meta(text_col: str | None, splitter) -> tuple[list[str], list[dict]]:
        if not text_col:
            return [], []
        texts = df[text_col].fillna("").astype(str).tolist()
        sentences: list[str] = []
        meta: list[dict] = []
        for idx, (text, title) in enumerate(zip(texts, titles_for_meta), start=1):
            parts = splitter(text)
            if not parts:
                continue
            sentences.extend(parts)
            meta.extend(
                {
                    "sentence": part,
                    "paper_title": title or f"Paper {idx}",
                    "paper_id": idx,
                }
                for part in parts
            )
        return sentences, meta

    abstract_sentences, abstract_meta = _build_sentences_and_meta(
        abstract_col, _split_sentences
    )
    title_sentences, title_meta = _build_sentences_and_meta(
        title_col, _split_sentences
    )
    keywords_terms, keywords_meta = _build_sentences_and_meta(
        keywords_col, _split_keywords
    )

    _ensure_dir(OUTPUT_DIR / "abstract")
    _ensure_dir(OUTPUT_DIR / "title")
    _ensure_dir(OUTPUT_DIR / "keywords")

    _save_json(OUTPUT_DIR / "abstract" / "sentences.json", abstract_sentences)
    _save_json(OUTPUT_DIR / "abstract" / "sentence_meta.json", abstract_meta)
    _save_json(OUTPUT_DIR / "title"    / "sentences.json", title_sentences)
    _save_json(OUTPUT_DIR / "title"    / "sentence_meta.json", title_meta)
    _save_json(OUTPUT_DIR / "keywords" / "sentences.json", keywords_terms)
    _save_json(OUTPUT_DIR / "keywords" / "sentence_meta.json", keywords_meta)

    df.to_csv(OUTPUT_DIR / "corpus.csv", index=False)

    return {
        "paper_count":             int(len(df)),
        "abstract_sentence_count": int(len(abstract_sentences)),
        "title_sentence_count":    int(len(title_sentences)),
        "keywords_term_count":     int(len(keywords_terms)),
        "detected_columns": {
            "abstract": abstract_col,
            "title": title_col,
            "keywords": keywords_col,
        },
        "columns":                 df.columns.tolist(),
        "sample_abstracts":        abstract_texts[:3],
        "filepath":                str(filepath),
    }


# ============================================================================
# TOOL 2 — run_bertopic_discovery
# ============================================================================

@tool
def run_bertopic_discovery(
    run_key: str,
    threshold: float = DISTANCE_THRESH,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    max_cluster_size: int = HDBSCAN_MAX_CLUSTER_SIZE,
    min_samples: int = HDBSCAN_MIN_SAMPLES,
    auto_optimize: bool = AUTO_OPTIMIZE_CLUSTERS,
    max_optimize_iters: int = OPTIMIZE_MAX_ITERS,
) -> dict:
    """
    Embed sentences, cluster with UMAP + HDBSCAN, extract evidence,
    and generate four Plotly charts.

    Saved artefacts
    ---------------
    emb.npy         : (N, D)   float32  L2-normalised embeddings
    sent_labels.npy : (N,)     int32    per-sentence cluster label  [BUG 1 FIX]
    summaries.json  : list of cluster dicts with evidence sentences
    optimization.json : list of optimization rounds and metrics

    Parameters
    ----------
    run_key   : str   — "abstract" or "title" or "keywords"
    threshold : float — legacy arg (ignored by HDBSCAN)
    min_cluster_size : int — HDBSCAN minimum cluster size
    max_cluster_size : int — HDBSCAN maximum cluster size
    min_samples : int — HDBSCAN min_samples
    auto_optimize : bool — run LLM-guided optimization loop
    max_optimize_iters : int — max optimization rounds after initial run

    Returns
    -------
    dict with keys:
        run_key, n_clusters, n_sentences, threshold,
        chart_paths, summaries_path, embeddings_path, optimization_path
    """
    if run_key not in RUN_CONFIGS:
        return {
            "run_key": run_key,
            "n_clusters": 0,
            "n_sentences": 0,
            "threshold": threshold,
            "chart_paths": {},
            "error": (
                f"Unsupported run_key: {run_key}. "
                f"Use one of: {', '.join(RUN_CONFIGS.keys())}."
            ),
        }

    rdir      = _run_dir(run_key)
    sent_path = OUTPUT_DIR / run_key / "sentences.json"
    if not sent_path.exists():
        return {
            "run_key": run_key,
            "n_clusters": 0,
            "n_sentences": 0,
            "threshold": threshold,
            "chart_paths": {},
            "error": (
                f"Missing sentences artifact: {sent_path}. "
                "Run load_scopus_csv first."
            ),
        }

    sentences = _load_json(sent_path)
    if not sentences:
        return {
            "run_key": run_key,
            "n_clusters": 0,
            "n_sentences": 0,
            "threshold": threshold,
            "chart_paths": {},
            "error": (
                f"No sentences/terms found for run_key={run_key}. "
                "Check that the corresponding source column exists in the CSV."
            ),
        }

    sentence_meta = _load_sentence_meta(run_key, sentences)

    emb_path = rdir / "emb.npy"
    embeddings = None
    if emb_path.exists():
        cached = np.load(str(emb_path))
        if cached.shape[0] == len(sentences):
            embeddings = cached

    if embeddings is None:
        embeddings = _embed(sentences)
        np.save(str(emb_path), embeddings)

    cluster_space = _umap_reduce(embeddings, UMAP_N_COMPONENTS_CLUSTER)
    umap_2d = _umap_reduce(embeddings, UMAP_N_COMPONENTS_VIZ)

    def _run_hdbscan(params: dict) -> tuple[list[int], dict]:
        labels_local = _cluster(
            cluster_space,
            min_cluster_size=int(params.get("min_cluster_size", HDBSCAN_MIN_CLUSTER_SIZE)),
            max_cluster_size=int(params.get("max_cluster_size", HDBSCAN_MAX_CLUSTER_SIZE)),
            min_samples=int(params.get("min_samples", HDBSCAN_MIN_SAMPLES)),
        ).tolist()
        return labels_local, _cluster_metrics(np.array(labels_local))

    current_params = {
        "min_cluster_size": int(min_cluster_size),
        "max_cluster_size": int(max_cluster_size),
        "min_samples": int(min_samples),
    }

    labels, metrics = _run_hdbscan(current_params)
    optimization_log = [
        {
            "round": 0,
            "params": current_params,
            "metrics": metrics,
        }
    ]

    best_score = _optimization_score(metrics)
    stable_rounds = 0

    seen_params = {(
        current_params["min_cluster_size"],
        current_params["max_cluster_size"],
        current_params["min_samples"],
    )}

    if bool(auto_optimize) and int(max_optimize_iters) > 0:
        for round_idx in range(1, int(max_optimize_iters) + 1):
            suggestion = _recommend_hdbscan_params(metrics, current_params)
            if suggestion.get("action") == "accept":
                optimization_log.append({
                    "round": round_idx,
                    "params": current_params,
                    "metrics": metrics,
                    "action": "accept",
                    "reasoning": suggestion.get("reasoning", ""),
                })
                break

            next_params = {
                "min_cluster_size": int(suggestion.get("min_cluster_size")),
                "max_cluster_size": int(suggestion.get("max_cluster_size")),
                "min_samples": int(suggestion.get("min_samples")),
            }
            next_key = (
                next_params["min_cluster_size"],
                next_params["max_cluster_size"],
                next_params["min_samples"],
            )
            if next_key in seen_params:
                optimization_log.append({
                    "round": round_idx,
                    "params": current_params,
                    "metrics": metrics,
                    "action": "stop",
                    "reasoning": "Repeated parameter set; stopping optimization.",
                })
                break

            labels, metrics = _run_hdbscan(next_params)
            optimization_log.append({
                "round": round_idx,
                "params": next_params,
                "metrics": metrics,
                "reasoning": suggestion.get("reasoning", ""),
            })
            current_params = next_params
            seen_params.add(next_key)

            score = _optimization_score(metrics)
            if score <= best_score + OPTIMIZE_MIN_IMPROVEMENT:
                stable_rounds += 1
            else:
                best_score = score
                stable_rounds = 0

            if _metrics_in_target(metrics):
                break

            if stable_rounds >= OPTIMIZE_STABLE_ROUNDS:
                break

    optimization_path = rdir / "optimization.json"
    _save_json(optimization_path, optimization_log)

    unique_ids = sorted(filter(lambda v: v != -1, set(labels)))

    # FIX BUG 1 — persist per-sentence label array so Tool 4 can build
    # correct cluster masks without any guesswork or scaffolding.
    np.save(str(rdir / "sent_labels.npy"), np.array(labels, dtype=np.int32))

    labels_arr = np.array(labels)

    if not unique_ids:
        _save_json(rdir / "summaries.json", [])
        return {
            "run_key":          run_key,
            "n_clusters":       0,
            "n_sentences":      int(len(sentences)),
            "threshold":        threshold,
            "min_cluster_size": int(current_params["min_cluster_size"]),
            "max_cluster_size": int(current_params["max_cluster_size"]),
            "min_samples":      int(current_params["min_samples"]),
            "chart_paths":      {},
            "summaries_path":   str(rdir / "summaries.json"),
            "embeddings_path":  str(rdir / "emb.npy"),
            "optimization_path": str(optimization_path),
            "error": "HDBSCAN produced no clusters (all points labeled as noise).",
        }

    def _cluster_summary(cid: int) -> dict:
        mask    = labels_arr == cid
        c_emb   = embeddings[mask]
        c_umap  = umap_2d[mask]
        c_sent  = list(np.array(sentences)[mask])
        ctroid  = _centroid(c_emb)
        top_idx = _top_k_indices(c_emb, ctroid, N_EVIDENCE)
        coords  = (
            c_umap.mean(axis=0)
            if c_umap.shape[0] > 0
            else np.zeros(UMAP_N_COMPONENTS_VIZ, dtype=np.float32)
        )
        paper_stats = _top_papers_for_mask(sentence_meta, mask, k=3)
        return {
            "cluster_id":  int(cid),
            "size":        int(mask.sum()),
            "cx":          float(coords[0]),
            "cy":          float(coords[1]),
            "evidence":    list(np.array(c_sent)[top_idx]),
            "paper_count": paper_stats.get("paper_count", 0),
            "top_papers":  paper_stats.get("top_papers", []),
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
        "run_key":          run_key,
        "n_clusters":       int(len(unique_ids)),
        "n_sentences":      int(len(sentences)),
        "threshold":        threshold,
        "min_cluster_size": int(current_params["min_cluster_size"]),
        "max_cluster_size": int(current_params["max_cluster_size"]),
        "min_samples":      int(current_params["min_samples"]),
        "chart_paths":      chart_paths,
        "summaries_path":   str(rdir / "summaries.json"),
        "embeddings_path":  str(rdir / "emb.npy"),
        "optimization_path": str(optimization_path),
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


_LABEL_JUDGE_PROMPT = PromptTemplate.from_template(
     """You are an expert label adjudicator. Choose the single best label from
the candidates below based on the evidence sentences.

Cluster ID    : {cluster_id}
Sentence count: {size}
Evidence sentences:
{evidence}

Candidate labels:
1) Mistral
    Label: {mistral_label}
    Category: {mistral_category}
    Confidence: {mistral_confidence}
    Reasoning: {mistral_reasoning}

2) Groq-Ollama
    Label: {groq_ollama_label}
    Category: {groq_ollama_category}
    Confidence: {groq_ollama_confidence}
    Reasoning: {groq_ollama_reasoning}

3) Groq-GPT
    Label: {groq_gpt_label}
    Category: {groq_gpt_category}
    Confidence: {groq_gpt_confidence}
    Reasoning: {groq_gpt_reasoning}

Rules:
- Choose exactly one of the three labels. Do not invent a new label.
- Pick the label that best matches the evidence and is most specific.
- If two are equally good, prefer the one with higher confidence.

Return RAW JSON with exactly these keys:
  best_label: string
  best_category: string
  chosen_source: string  # one of: mistral, groq_ollama, groq_gpt
  best_reasoning: string

Respond with RAW JSON only.
"""
)


@tool
def label_topics_with_llm(run_key: str) -> dict:
    """
    Label each cluster with Mistral only (default Phase 2 labeling pass).

    Parameters
    ----------
    run_key : str — "abstract" or "title" or "keywords"

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
            "groq_ollama_label":  "",
            "groq_ollama_category": "",
            "groq_ollama_confidence": 0.0,
            "groq_ollama_reasoning": "",
            "groq_ollama_niche":  False,
            "groq_gpt_label":     "",
            "groq_gpt_category":  "",
            "groq_gpt_confidence": 0.0,
            "groq_gpt_reasoning": "",
            "groq_gpt_niche":     False,
            "verification_done":  False,
            "verification_done_ollama": False,
            "verification_done_gpt": False,
            "verification_note":  (
                "Run VERIFY in Phase 2 to compare with Groq-Ollama and Groq-GPT labels."
            ),
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
            "groq_ollama_label": r.get("groq_ollama_label", r.get("groq_label", "")),
            "groq_gpt_label": r.get("groq_gpt_label", ""),
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
        "groq_enabled":      _groq_ollama_enabled() and _groq_gpt_enabled(),
        "mode_note":         "Single-model labeling complete (Mistral). Send VERIFY in Phase 2 to run Groq-Ollama and Groq-GPT verification.",
        "labels_preview":    preview,
    }


@tool
def verify_topic_labels_with_groq(run_key: str) -> dict:
    """
    Run Groq topic labeling for already-labeled topics and append comparison fields
    into labels.json so UI review table can show Mistral vs Groq-Ollama vs Groq-GPT labels,
    plus an adjudicated best label when GROQ_JUDGE_MODEL_NAME is configured.

    Parameters
    ----------
    run_key : str — "abstract" or "title" or "keywords"

    Returns
    -------
    dict with keys:
        run_key, labels_path, verification_path, verified_count, labels_preview
    """
    rdir          = _run_dir(run_key)
    labels_path   = rdir / "labels.json"
    summaries_path = rdir / "summaries.json"

    if not _groq_ollama_enabled() or not _groq_gpt_enabled():
        return {
            "run_key": run_key,
            "labels_path": str(labels_path),
            "verified_count": 0,
            "labels_preview": [],
            "error": (
                "GROQ_API_KEY or Groq model config is missing, or langchain-groq is unavailable. "
                "Set GROQ_API_KEY and GROQ_GPT_MODEL_NAME (and optionally GROQ_OLLAMA_MODEL_NAME) "
                "and install requirements to use VERIFY."
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

    chain_groq_ollama = _LABEL_PROMPT | _llm_groq(GROQ_OLLAMA_MODEL_NAME) | JsonOutputParser()
    chain_groq_gpt = _LABEL_PROMPT | _llm_groq(GROQ_GPT_MODEL_NAME) | JsonOutputParser()
    chain_judge = (
        _LABEL_JUDGE_PROMPT | _llm_groq(GROQ_JUDGE_MODEL_NAME) | JsonOutputParser()
        if _groq_judge_enabled()
        else None
    )

    def _evidence_block(summary: dict) -> str:
        return "\n".join(
            f"  {i+1}. {s}"
            for i, s in enumerate(summary.get("evidence", []))
        )

    def _label_with_groq(row: dict) -> tuple[int, dict, dict]:
        cid = int(row.get("cluster_id", -1))
        summary = summary_by_id[cid]
        payload = {
            "cluster_id": summary["cluster_id"],
            "size":       summary["size"],
            "evidence":   _evidence_block(summary),
        }
        groq_ollama = _invoke_with_retries(lambda: chain_groq_ollama.invoke(payload))
        groq_gpt = _invoke_with_retries(lambda: chain_groq_gpt.invoke(payload))
        return cid, groq_ollama, groq_gpt

    groq_pairs = list(map(_label_with_groq, target_rows))
    groq_ollama_by_id = {cid: data for cid, data, _ in groq_pairs}
    groq_gpt_by_id = {cid: data for cid, _, data in groq_pairs}

    def _judge_label(row: dict) -> tuple[int, dict]:
        if chain_judge is None:
            return int(row.get("cluster_id", -1)), {}
        cid = int(row.get("cluster_id", -1))
        summary = summary_by_id[cid]
        groq_ollama = groq_ollama_by_id.get(cid, {})
        groq_gpt = groq_gpt_by_id.get(cid, {})
        payload = {
            "cluster_id": summary.get("cluster_id"),
            "size": summary.get("size"),
            "evidence": _evidence_block(summary),
            "mistral_label": str(row.get("mistral_label") or row.get("label", "")).strip(),
            "mistral_category": str(row.get("mistral_category") or row.get("category", "")).strip(),
            "mistral_confidence": _to_float(row.get("mistral_confidence", row.get("confidence", 0.0)), 0.0),
            "mistral_reasoning": str(row.get("mistral_reasoning") or row.get("reasoning", "")).strip(),
            "groq_ollama_label": str(groq_ollama.get("label", "")).strip(),
            "groq_ollama_category": str(groq_ollama.get("category", "")).strip(),
            "groq_ollama_confidence": _to_float(groq_ollama.get("confidence"), 0.0),
            "groq_ollama_reasoning": str(groq_ollama.get("reasoning", "")).strip(),
            "groq_gpt_label": str(groq_gpt.get("label", "")).strip(),
            "groq_gpt_category": str(groq_gpt.get("category", "")).strip(),
            "groq_gpt_confidence": _to_float(groq_gpt.get("confidence"), 0.0),
            "groq_gpt_reasoning": str(groq_gpt.get("reasoning", "")).strip(),
        }
        try:
            result = _invoke_with_retries(lambda: chain_judge.invoke(payload))
        except Exception:
            result = {}
        return cid, result

    judge_pairs = list(map(_judge_label, target_rows)) if chain_judge else []
    judge_by_id = {cid: data for cid, data in judge_pairs}

    def _merge_row(row: dict) -> dict:
        cid = int(row.get("cluster_id", -1))
        groq_ollama = groq_ollama_by_id.get(cid, {})
        groq_gpt = groq_gpt_by_id.get(cid, {})
        adjudicated = judge_by_id.get(cid, {})
        has_groq_ollama = bool(groq_ollama)
        has_groq_gpt = bool(groq_gpt)
        mistral_label = str(row.get("mistral_label") or row.get("label", "")).strip()
        groq_ollama_label = str(groq_ollama.get("label", "")).strip()
        groq_gpt_label = str(groq_gpt.get("label", "")).strip()
        adjudicated_label = str(adjudicated.get("best_label", "")).strip()
        is_agreement = (
            all([mistral_label, groq_ollama_label, groq_gpt_label])
            and mistral_label.lower() == groq_ollama_label.lower()
            and mistral_label.lower() == groq_gpt_label.lower()
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
            "groq_label":         groq_ollama_label,
            "groq_category":      groq_ollama.get("category", ""),
            "groq_confidence":    _to_float(groq_ollama.get("confidence"), 0.0),
            "groq_reasoning":     groq_ollama.get("reasoning", ""),
            "groq_niche":         bool(groq_ollama.get("niche", False)),
            "groq_ollama_label":  groq_ollama_label,
            "groq_ollama_category": groq_ollama.get("category", ""),
            "groq_ollama_confidence": _to_float(groq_ollama.get("confidence"), 0.0),
            "groq_ollama_reasoning": groq_ollama.get("reasoning", ""),
            "groq_ollama_niche":  bool(groq_ollama.get("niche", False)),
            "groq_gpt_label":     groq_gpt_label,
            "groq_gpt_category":  groq_gpt.get("category", ""),
            "groq_gpt_confidence": _to_float(groq_gpt.get("confidence"), 0.0),
            "groq_gpt_reasoning": groq_gpt.get("reasoning", ""),
            "groq_gpt_niche":     bool(groq_gpt.get("niche", False)),
            "adjudicated_label":  adjudicated_label,
            "adjudicated_category": str(adjudicated.get("best_category", "")).strip(),
            "adjudicated_reasoning": str(adjudicated.get("best_reasoning", "")).strip(),
            "adjudicated_source": str(adjudicated.get("chosen_source", "")).strip(),
            "adjudication_done":  bool(adjudicated_label),
            "adjudication_note": (
                "Adjudicated label available."
                if adjudicated_label
                else "Adjudication unavailable for this topic."
            ),
            "verification_done":  has_groq_ollama and has_groq_gpt,
            "verification_done_ollama": has_groq_ollama,
            "verification_done_gpt": has_groq_gpt,
            "verification_note": (
                "Mistral, Groq-Ollama, and Groq-GPT labels match."
                if is_agreement
                else "Model labels differ. Review before approval."
            )
            if has_groq_ollama and has_groq_gpt
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
            "groq_ollama_label": r.get("groq_ollama_label", r.get("groq_label", "")),
            "groq_gpt_label": r.get("groq_gpt_label", ""),
            "adjudicated_label": r.get("adjudicated_label", ""),
            "verification_note": r.get("verification_note", ""),
        },
        verified_rows[:MAX_TOOL_RETURN_PREVIEW],
    ))

    verified_count = sum(
        1
        for row in verified_rows
        if row.get("groq_ollama_label") and row.get("groq_gpt_label")
    )

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
    run_key   : str  — "abstract" or "title" or "keywords"
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
    run_key : str — "abstract" or "title" or "keywords"

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


@tool
def verify_taxonomy_mapping_with_groq(run_key: str) -> dict:
    """
    Run Groq validation for PAJAIS taxonomy mappings and persist side-by-side
    Mistral/Groq mapping fields for each theme.

    Parameters
    ----------
    run_key : str — "abstract" or "title" or "keywords"

    Returns
    -------
    dict with keys:
        run_key, taxonomy_path, verification_path,
        verified_count, mapping_preview
    """
    if not _groq_ollama_enabled():
        return {
            "run_key": run_key,
            "taxonomy_path": str(_run_dir(run_key) / "taxonomy_map.json"),
            "verified_count": 0,
            "mapping_preview": [],
            "error": (
                "GROQ_API_KEY is missing or langchain-groq is unavailable. "
                "Set GROQ_API_KEY and install requirements to use VERIFY."
            ),
        }

    rdir          = _run_dir(run_key)
    themes_path   = rdir / "themes.json"
    taxonomy_path = rdir / "taxonomy_map.json"

    if not themes_path.exists():
        return {
            "run_key": run_key,
            "taxonomy_path": str(taxonomy_path),
            "verified_count": 0,
            "mapping_preview": [],
            "error": (
                f"Missing themes artifact: {themes_path}. "
                "Run consolidate_into_themes first."
            ),
        }

    if not taxonomy_path.exists():
        return {
            "run_key": run_key,
            "taxonomy_path": str(taxonomy_path),
            "verified_count": 0,
            "mapping_preview": [],
            "error": (
                f"Missing taxonomy artifact: {taxonomy_path}. "
                "Run compare_with_taxonomy first."
            ),
        }

    themes       = _load_json(themes_path)
    taxonomy_map = _load_json(taxonomy_path)
    taxonomy_str = "\n".join(f"  - {cat}" for cat in PAJAIS_TAXONOMY)

    chain_groq = _TAXONOMY_PROMPT | _llm_groq(GROQ_OLLAMA_MODEL_NAME) | JsonOutputParser()

    def _map_theme_with_groq(theme: dict) -> dict:
        return _invoke_with_retries(lambda: chain_groq.invoke({
            "taxonomy":   taxonomy_str,
            "theme_name": theme["theme_name"],
            "evidence":   " | ".join(theme.get("evidence", [])[:3]),
        }))

    groq_maps = list(map(_map_theme_with_groq, themes))
    groq_by_theme = {
        str(item.get("theme_name", "")).strip(): item
        for item in groq_maps
    }

    def _merge_mappings(mistral_row: dict) -> dict:
        theme_name = str(mistral_row.get("theme_name", "")).strip()
        groq_row = groq_by_theme.get(theme_name, {})
        groq_match = str(groq_row.get("pajais_match", "")).strip()
        mistral_match = str(mistral_row.get("pajais_match", "")).strip()
        is_same = bool(groq_match) and (groq_match.lower() == mistral_match.lower())

        return {
            **mistral_row,
            "mistral_pajais_match": mistral_match,
            "mistral_confidence": _to_float(
                mistral_row.get("mistral_confidence", mistral_row.get("confidence", 0.0)),
                0.0,
            ),
            "mistral_reasoning": str(
                mistral_row.get("mistral_reasoning", mistral_row.get("reasoning", ""))
            ),
            "mistral_is_novel": bool(
                mistral_row.get("mistral_is_novel", mistral_row.get("is_novel", False))
            ),
            "groq_pajais_match": groq_match,
            "groq_confidence": _to_float(groq_row.get("confidence"), 0.0),
            "groq_reasoning": str(groq_row.get("reasoning", "")),
            "groq_is_novel": bool(groq_row.get("is_novel", False)),
            "taxonomy_verification_done": bool(groq_row),
            "taxonomy_verification_note": (
                "Mistral and Groq taxonomy mapping match."
                if is_same
                else "Mistral and Groq taxonomy mapping differ."
            ) if groq_row else "Groq taxonomy mapping unavailable for this theme.",
        }

    merged_rows = list(map(_merge_mappings, taxonomy_map))
    verification_path = rdir / "taxonomy_verification.json"
    _save_json(taxonomy_path, merged_rows)
    _save_json(verification_path, merged_rows)

    preview = list(map(
        lambda row: {
            "theme_name": row.get("theme_name", ""),
            "mistral_pajais_match": row.get("mistral_pajais_match", row.get("pajais_match", "")),
            "groq_pajais_match": row.get("groq_pajais_match", ""),
            "taxonomy_verification_note": row.get("taxonomy_verification_note", ""),
        },
        merged_rows[:MAX_TOOL_RETURN_PREVIEW],
    ))

    verified_count = sum(1 for row in merged_rows if row.get("groq_pajais_match"))

    return {
        "run_key": run_key,
        "taxonomy_path": str(taxonomy_path),
        "verification_path": str(verification_path),
        "verified_count": int(verified_count),
        "mapped_count": int(len(merged_rows)),
        "mapping_preview": preview,
    }


# ============================================================================
# TOOL 6 — generate_comparison_csv
# ============================================================================

@tool
def generate_comparison_csv() -> dict:
    """
    Side-by-side comparison of abstract/title/keywords theme mappings.

    Each run is optional. Missing runs produce empty columns.

    Returns
    -------
    dict with keys:
        csv_path, row_count, columns, preview (list of dicts)
    """
    abstract_path = OUTPUT_DIR / "abstract" / "taxonomy_map.json"
    title_path    = OUTPUT_DIR / "title"    / "taxonomy_map.json"
    keywords_path = OUTPUT_DIR / "keywords" / "taxonomy_map.json"

    abstract_map = _load_json(abstract_path) if abstract_path.exists() else []
    title_map    = _load_json(title_path) if title_path.exists() else []
    keywords_map = _load_json(keywords_path) if keywords_path.exists() else []

    if not (abstract_map or title_map or keywords_map):
        return {
            "csv_path": str(OUTPUT_DIR / "comparison.csv"),
            "row_count": 0,
            "columns": [],
            "preview": [],
            "error": (
                "No taxonomy_map.json files found for abstract/title/keywords. "
                "Run compare_with_taxonomy for at least one run first."
            ),
        }

    def _row(a_theme: dict | None, t_theme: dict | None, k_theme: dict | None) -> dict:
        return {
            "Abstract Theme":      a_theme.get("theme_name",   "") if a_theme else "",
            "Abstract PAJAIS":     a_theme.get("pajais_match",  "") if a_theme else "",
            "Abstract Confidence": a_theme.get("confidence",    0) if a_theme else 0,
            "Abstract Novel":      a_theme.get("is_novel",     False) if a_theme else False,
            "Title Theme":         t_theme.get("theme_name",   "") if t_theme else "",
            "Title PAJAIS":        t_theme.get("pajais_match",  "") if t_theme else "",
            "Title Confidence":    t_theme.get("confidence",    0)  if t_theme else 0,
            "Title Novel":         t_theme.get("is_novel",     False) if t_theme else False,
            "Keywords Theme":      k_theme.get("theme_name",   "") if k_theme else "",
            "Keywords PAJAIS":     k_theme.get("pajais_match",  "") if k_theme else "",
            "Keywords Confidence": k_theme.get("confidence",    0) if k_theme else 0,
            "Keywords Novel":      k_theme.get("is_novel",     False) if k_theme else False,
        }

    max_len  = max(len(abstract_map), len(title_map), len(keywords_map), 1)
    padded_a = abstract_map + [{}] * (max_len - len(abstract_map))
    padded_t = title_map    + [{}] * (max_len - len(title_map))
    padded_k = keywords_map + [{}] * (max_len - len(keywords_map))

    rows = list(map(_row, padded_a, padded_t, padded_k))
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
Structure: (1) methodology overview, (2) major themes found across runs,
(3) PAJAIS alignment, (4) novel contributions, (5) limitations.

Use formal academic English. Do NOT use bullet points.

Abstract themes & taxonomy:
{abstract_themes}

Title themes & taxonomy:
{title_themes}

Keywords themes & taxonomy:
{keywords_themes}

Respond with plain text only.
"""
)


@tool
def export_narrative(run_key: str) -> dict:
    """
    Generate a 500-word academic narrative and save to narrative.txt.

    Parameters
    ----------
    run_key : str — "abstract" or "title" or "keywords" (primary source)

    Returns
    -------
    dict with keys:
        narrative_path, word_count, preview (first 300 chars)
    """
    rdir          = _run_dir(run_key)
    abstract_path = OUTPUT_DIR / "abstract" / "taxonomy_map.json"
    title_path    = OUTPUT_DIR / "title" / "taxonomy_map.json"
    keywords_path = OUTPUT_DIR / "keywords" / "taxonomy_map.json"

    abstract_map = _load_json(abstract_path) if abstract_path.exists() else []
    title_map    = _load_json(title_path) if title_path.exists() else []
    keywords_map = _load_json(keywords_path) if keywords_path.exists() else []

    if not (abstract_map or title_map or keywords_map):
        return {
            "narrative_path": str(rdir / "narrative.txt"),
            "word_count": 0,
            "preview": "",
            "error": (
                "No taxonomy mappings found for abstract/title/keywords. "
                "Run compare_with_taxonomy before export_narrative."
            ),
        }

    def _theme_summary(t: dict) -> str:
        return (
            f"  - {t.get('theme_name','?')} -> {t.get('pajais_match','?')} "
            f"(conf={t.get('confidence',0):.2f}, novel={t.get('is_novel',False)})"
        )

    abstract_str = "\n".join(map(_theme_summary, abstract_map))
    title_str    = "\n".join(map(_theme_summary, title_map)) or "Not run."
    keywords_str = "\n".join(map(_theme_summary, keywords_map)) or "Not run."

    chain    = _NARRATIVE_PROMPT | _llm()
    response = _invoke_with_retries(lambda: chain.invoke({
        "abstract_themes": abstract_str,
        "title_themes":    title_str,
        "keywords_themes": keywords_str,
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
    verify_taxonomy_mapping_with_groq,
    generate_comparison_csv,
    export_narrative,
]
