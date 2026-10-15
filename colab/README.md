# Colab mid-scale track

The laptop fits 0.8B iteration and 2B verification. Anything bigger (9B
frozen probes, longer stage-3 runs, BTX at scale) goes here or to Kaggle.

## Colab (interactive, this track)

Best for: 2B distill/MoCM verification, 9B frozen probes, longer single runs
with inspection. Free tier: T4-class GPU, ~12GB RAM (high-RAM setting helps
the 9B frozen probe), 12 h sessions, opaque weekly quota, ephemeral disk.

1. One-time setup: accept the Qwen license on Hugging Face, store an access
   token in Colab Secrets as `HF_TOKEN`.
2. Pack the repo (same tarball recipe as `kaggle/README.md`) and upload
   `moeme-repo.tar.gz` to Drive.
3. Open `colab/mocm_midscale.ipynb`, set runtime to GPU, run top to bottom.
   Outputs land in Drive under `moeme_colab/`.
4. Copy `results/*.json` back into this repo's `results/` with a `colab_` prefix
   and note the provenance (cell 4 does this for the eval).

## Kaggle (batch, existing track)

Best for: scheduled unattended runs with transparent-ish quota (30 GPU-h/week,
2xT4, 30GB RAM, 9 h sessions). See `kaggle/README.md`. E15 ran there.

## Neither trains 27B

27B bf16 is 56GB; int8 is 28GB. No free tier holds training. Free tiers buy
verification at 2-9B and frozen probes — which is exactly what this track runs.
The 27B question is answered by measurement (range-fetched layers,
`experiments/mocm_bytes.py`) plus residency benching, not by training it.
