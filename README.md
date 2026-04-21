# BERTopic Thematic Analysis Agent

Production Gradio dashboard for phase-gated thematic analysis of Scopus exports.

The system combines:

- Sentence-BERT embeddings (`all-MiniLM-L6-v2`)
- Agglomerative clustering for topic discovery
- LangGraph ReAct orchestration
- Mistral topic labeling by default
- Optional Phase 2 `VERIFY` command to add Groq labels for side-by-side comparison
- Mistral for taxonomy mapping and narrative generation

## Highlights

- End-to-end pipeline aligned with Braun and Clarke thematic analysis phases
- Researcher-in-the-loop review table (approve, rename, reason)
- Automatic chart generation and in-UI rendering
- Taxonomy mapping against PAJAIS categories
- Exports for reproducible reporting (`csv`, `json`, `txt`, `html`, `npy`)

## Repository Layout

- `app.py` - Gradio UI, event wiring, chart embedding, downloads panel
- `agent.py` - phase-gated LangGraph agent and state transitions
- `tools.py` - eight analysis tools used by the agent
- `requirements.txt` - Python dependencies
- `uploads/` - persisted uploaded CSV files
- `outputs/` - generated artifacts and charts

## Requirements

- Python 3.11 or 3.12 recommended
- Python 3.14 can run, but LangChain may emit pydantic-v1 compatibility warnings

## Setup

```bash
cd "/home/atharv/Desktop/Thematic Analysis"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables

Set in the same shell where you run the app:

```bash
export MISTRAL_API_KEY="your-mistral-api-key"
export GROQ_API_KEY="your-groq-api-key"       # enables Phase 2 VERIFY (Groq label comparison)
export HF_TOKEN="your-hf-token"  # optional but recommended for model downloads

# Optional: Groq model for VERIFY command
export GROQ_MODEL_NAME="llama-3.3-70b-versatile"
```

## Run

```bash
cd "/home/atharv/Desktop/Thematic Analysis"
source venv/bin/activate
python app.py
```

Open the URL shown in the terminal, usually:

- `http://0.0.0.0:7860`
- `http://127.0.0.1:7860`

## Input CSV Format

Required columns:

- `Title`
- `Abstract`

Additional columns are allowed.

## Analysis Workflow

1. Upload a CSV in the Data Input panel.
2. The app auto-triggers Phase 1 context once upload succeeds.
3. In chat, start discovery with `run abstract`.
4. At Phase 2, optionally type `VERIFY` in chat to add Groq labels.
5. Review generated topics in the Review tab (`Mistral Label` and `Groq Label`).
6. Edit `Approve`, `Rename To`, and `Reasoning`, then click Submit Review.
7. Continue phase-by-phase through theme consolidation, review, naming, taxonomy mapping, and report generation.
8. Use Charts and Downloads tabs for visual and file outputs.

## Useful Chat Prompts

- `run abstract`
- `VERIFY`
- `run title`
- `show topics`
- `export results`
- `generate report`

The agent is robust to natural-language variants, but short explicit prompts are most reliable.

## Tools Implemented in `tools.py`

1. `load_scopus_csv(filepath)`
2. `run_bertopic_discovery(run_key, threshold)`
3. `label_topics_with_llm(run_key)`
4. `verify_topic_labels_with_groq(run_key)`
5. `consolidate_into_themes(run_key, theme_map)`
6. `compare_with_taxonomy(run_key)`
7. `generate_comparison_csv()`
8. `export_narrative(run_key)`

## Output Artifacts

Common outputs include:

- `outputs/corpus.csv`
- `outputs/comparison.csv`
- `outputs/abstract/sentences.json`
- `outputs/abstract/emb.npy`
- `outputs/abstract/sent_labels.npy`
- `outputs/abstract/summaries.json`
- `outputs/abstract/labels.json`
- `outputs/abstract/themes.json`
- `outputs/abstract/taxonomy_map.json`
- `outputs/abstract/narrative.txt`
- `outputs/abstract/intertopic.html`
- `outputs/abstract/topwords.html`
- `outputs/abstract/hierarchy.html`
- `outputs/abstract/heatmap.html`

Equivalent files are produced under `outputs/title/` when title-run analysis is executed.

## Charts in UI

Charts are rendered from generated HTML files in `outputs/`.

Current implementation details:

- Uses Gradio 6 file route: `/gradio_api/file=...`
- URL-encodes file paths to handle spaces in directories
- Sets `allowed_paths` in `demo.launch(...)` to allow serving `outputs/`
- Auto-refreshes chart panel after agent responses

## Troubleshooting

### Charts not visible with {"detail":"Not Found"}

- Ensure you are on the latest `app.py` and restart the server.
- Hard refresh browser (`Ctrl+Shift+R`).
- Confirm chart files exist under `outputs/<run_key>/`.

### MISTRAL key missing

If the UI or terminal reports missing API key:

```bash
export MISTRAL_API_KEY="your-mistral-api-key"
```

Restart the app after setting it.

### GROQ key missing

Without `GROQ_API_KEY`, topic labeling still runs with Mistral, but `VERIFY` cannot run.

```bash
export GROQ_API_KEY="your-groq-api-key"
```

### Hugging Face unauthenticated warning

Set `HF_TOKEN` to improve rate limits and download reliability.

### Temporary Mistral outages (503 / unreachable_backend)

The code includes retries and bounded backoff. If all retries fail, retry your last command after 30 to 60 seconds.

### Slow runtime on large corpora

Expected for large datasets due to embedding and LLM labeling costs. Labeling is already capped to reduce transcript and latency pressure.

## Security Notes

- Never commit API keys.
- Rotate keys immediately if they appear in logs, screenshots, or chat history.

## Citation Context

Methodological framing and implementation are inspired by:

- Braun and Clarke (2006) reflexive thematic analysis
- Grootendorst (2022) BERTopic
- Sentence-BERT literature for semantic embeddings
