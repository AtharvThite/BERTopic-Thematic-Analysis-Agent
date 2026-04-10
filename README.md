# BERTopic Thematic Analysis Agent

A Gradio-based research dashboard that runs a phase-gated thematic analysis pipeline using BERTopic-style clustering, LangGraph orchestration, and Mistral for labeling and narrative generation.

## What this project does

- Loads Scopus-style CSV files.
- Splits and preprocesses `Abstract` and `Title` text into sentence corpora.
- Generates embeddings and clusters sentences into topics.
- Labels clusters with an LLM.
- Lets you review and approve topics in a table.
- Consolidates approved topics into themes.
- Maps themes to a PAJAIS taxonomy.
- Exports comparison CSVs and narrative outputs.

## Project structure

- `app.py`: Gradio UI and event wiring.
- `agent.py`: LangGraph ReAct agent wrapper with phase gating.
- `tools.py`: Tool functions for each pipeline stage.
- `requirements.txt`: Python dependencies.
- `uploads/`: Persisted uploaded CSV files.
- `outputs/`: Generated artifacts.

## Prerequisites

- Python 3.11 or 3.12 is recommended.
- Python 3.14 may run, but you can see compatibility warnings from LangChain internals.

## Installation

```bash
cd "/home/atharv/Desktop/Thematic Analysis"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Environment variables

Set these in the same terminal where you run the app:

```bash
export MISTRAL_API_KEY="your-mistral-api-key"
# optional, improves Hugging Face model download reliability and limits
export HF_TOKEN="your-hf-token"
```

## Run locally

```bash
cd "/home/atharv/Desktop/Thematic Analysis"
source venv/bin/activate
python app.py
```

Open the URL printed in terminal (usually `http://0.0.0.0:7860` or `http://127.0.0.1:7860`).

## CSV format

Expected columns:

- `Title`
- `Abstract`

Other columns are allowed, but these two are required for the main flow.

## UI workflow

1. Upload a CSV from the Data Input card.
2. In chat, run: `run abstract`
3. Wait for Phase 2 to complete and review table to populate.
4. In the Review tab:
- Set `Approve` for accepted topics.
- Optionally edit `Rename To` and `Reasoning`.
- Click `Submit Review`.
5. Continue through phases in chat:
- `confirm themes`
- `approve themes`
- `run taxonomy`
- `generate report`
6. Download files from the Downloads tab.

## Generated outputs

Typical files under `outputs/`:

- `outputs/corpus.csv`
- `outputs/abstract/sentences.json`
- `outputs/abstract/emb.npy`
- `outputs/abstract/sent_labels.npy`
- `outputs/abstract/summaries.json`
- `outputs/abstract/labels.json`
- `outputs/abstract/themes.json`
- `outputs/abstract/taxonomy_map.json`
- `outputs/abstract/narrative.txt`
- `outputs/comparison.csv`
- Chart HTML files such as `intertopic.html`, `topwords.html`, `hierarchy.html`, `heatmap.html`

## Troubleshooting

### Missing MISTRAL key

If you see:

- `MISTRAL_API_KEY is not set`

Then set the key and restart the app in the same shell.

### Hugging Face warning

If you see unauthenticated HF warnings, set `HF_TOKEN` for better rate limits and download behavior.

### Review table Approve checkbox not clickable

Restart after pulling latest changes. The table is configured to keep `Approve` as interactive boolean cells.

### `summaries.json` not found

Re-run from Phase 1/2 after uploading a file. Output paths are now anchored to the project directory.

### Long wait on `run abstract`

Large datasets can be slow. Labeling is capped to top clusters and LLM calls use fail-fast timeout settings, but runtime still depends on corpus size and API latency.

## Notes

- Do not commit API keys to source files.
- Rotate keys if they were exposed in terminal logs or chats.
