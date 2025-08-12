# DisasterHybrid — Hybrid CV + NLP Demo

## What this repo contains
- Lightweight UNet segmentation to detect damaged/flooded areas in satellite/drone images.
- DistilBERT-based tweet classifier for live or simulated disaster tweets.
- Streamlit dark-themed dashboard (neon-green accents) that shows:
  - Image upload → segmentation overlay
  - Tweet stream (offline by default) + single-tweet classifier
  - Demo map view

## Quickstart (local)
1. Create virtual env & install:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt

