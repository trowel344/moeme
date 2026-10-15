# Lightning CLI runbook (machine-side)

CLI installed (`pip install lightning-sdk`, binary at `~/.local/bin/lightning`).
Catalog verified: A100, H100/H200, B200, L4, L40S, T4 all listed.

## Step 0 — you do this (interactive browser login, 1 min)

```bash
export PATH="$HOME/.local/bin:$PATH"
lightning login
lightning user  # confirm identity / teamspace
```

Nothing below works until this is done, and nothing below spends anything
until a Studio is *started* — creating is free, running bills.

## Step 1 — start the right machine per job, stop when done

```bash
# L4 track (9B program, lightning/L4-31h.md), ~10 h total across sessions:
lightning studio start --name mocm-l4 --machine L4 --create
# ... run jobs, then ALWAYS:
lightning studio stop --name mocm-l4

# A100 track (27B oracle table + dry runs, lightning/A100-40-10h.md):
lightning studio start --name mocm-a100 --machine A100 --create
lightning studio stop --name mocm-a100

# 80GB gate (lightning/README.md month-one run): A100 is 40GB or 80GB
# depending on teamspace allocation — check `nvidia-smi` inside the Studio
# first; if 40GB, run the 4-bit fallback command from the README.
```

Cost hygiene (meters run while started, not while created):
- `lightning studio list` shows what's running. Stopped studios keep disk.
- `lightning studio stop` after EVERY session. Set a phone timer.
- `lightning studio delete` when a track is fully done.
- Never leave a session attached overnight "to keep caches warm".

## Step 2 — inside the Studio (per fresh start)

```bash
nvidia-smi  # confirm GPU + VRAM (40 vs 80 decides the command)
free -g; df -h / | tail -1
pip install -q transformers accelerate bitsandbytes sentencepiece datasets huggingface_hub
pip install -q causal-conv1d  # ~10 min build if no wheel; Qwen GDN needs it
huggingface-cli login
tar -xzf moeme-repo.tar.gz && cd moeme && python3 -m pytest tests/ -q  # sanity
```

Upload the repo tarball once per Studio (`lightning studio cp` or the web UI);
keep `results/` syncing back the same way after each job.

## Step 3 — job commands (copy-paste per track file)

- L4: `lightning/L4-31h.md` jobs 1-3 in order (frozen oracle → depth → distill).
- A100-40: `lightning/A100-40-10h.md` (oracle table first, then dry-runs).
- A100-80: `lightning/README.md` month-one gate, only after the A100-40
  dry-run prices it exactly.

## Quota/blackout notes

- Free-tier GPU availability varies by hour; if A100 shows unavailable, run
  the L4 track (always useful) and retry off-peak.
- `--interruptible` (spot) is cheaper but can preempt mid-run: only use with
  `--resume` jobs (all PoC drivers support it) and `--max-seconds` caps.
- The £21.68/8h figure is paid credits stacked on free hours (13 h total):
  confirm the current price in-teamspace before buying; run free hours first.
