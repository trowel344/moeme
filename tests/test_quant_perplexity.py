"""
Tests for the quantisation-damage probe (experiments/quant_perplexity.py).

The numbers this script emits are quoted as the answer to "is the small
quantisation worse?", so the two ways it can lie are worth pinning offline:

1. reading a *killed* run as a number. The 21 GB MoE's Q4_K_M run was OOM-killed
   before it printed anything; a parser that defaulted a missing estimate to 0
   would have made the smallest quant look best.
2. reading the printed series as per-chunk perplexities. llama-perplexity prints a
   CUMULATIVE value after each chunk (the last one equals the final estimate
   exactly), so differencing it would invent per-chunk numbers below 1 -- a
   probability above certainty, which is how the mistake shows up.
3. desyncing on GGUF metadata and reporting "?" for a file that states its
   quantisation. This is not hypothetical: it happened, on both real artifacts,
   because the tokenizer's token array precedes `general.file_type`.
"""

import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

import quant_perplexity  # noqa: E402
from quant_perplexity import (  # noqa: E402
    build_corpus,
    fmt_bpw,
    gguf_metadata,
    parse_ppl,
    window_note,
)

# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

REAL_LOG = """\
0.00.930.529 W llama_model_loader: tensor overrides to CPU are used with mmap enabled
0.15.137.062 I perplexity: tokenization took 104.695 ms
0.15.137.138 I perplexity: calculating perplexity over 8 chunks, n_ctx=512, batch_size=2048, n_seq=4
1.29.771.068 I perplexity: 74.63 seconds per pass - ETA 2.48 minutes
[1]8.3929,[2]12.2213,[3]6.2730,[4]4.8010,[5]4.9734,[6]5.4764,[7]5.9155,[8]6.5966,
2.46.871.431 I Final estimate: PPL = 6.5966 +/- 0.38006
"""


def test_parse_ppl_reads_the_final_estimate():
    got = parse_ppl(REAL_LOG)
    assert got["ppl"] == 6.5966
    assert got["stderr"] == 0.38006
    assert got["chunks_reported"] == 8


def test_printed_series_is_cumulative_not_per_chunk():
    """The last printed value IS the final estimate -- that is the tell."""
    got = parse_ppl(REAL_LOG)
    assert got["cumulative_series"][-1] == got["ppl"]
    assert got["cumulative_series"] == [8.3929, 12.2213, 6.2730, 4.8010,
                                        4.9734, 5.4764, 5.9155, 6.5966]


def test_a_killed_run_has_no_number_rather_than_a_zero():
    """The Q4_K_M 35B run died during load; it must not read as 0.0."""
    killed = ("0.00.930.529 W llama_model_loader: tensor overrides to CPU are used "
              "with mmap enabled - consider using --no-mmap for better performance\n")
    got = parse_ppl(killed)
    assert got["ppl"] is None
    assert got["cumulative_series"] == []
    assert got["final_estimate_line"] is None


def test_parse_ppl_tolerates_a_negative_or_scientific_estimate():
    # Guard against a regex that only matches the shape seen so far.
    got = parse_ppl("I Final estimate: PPL = 12.5 +/- 3.25\n")
    assert (got["ppl"], got["stderr"]) == (12.5, 3.25)


# ---------------------------------------------------------------------------
# the scored window: what the number is actually a measurement of
# ---------------------------------------------------------------------------

def test_parse_ppl_records_the_scored_window():
    """`chunks: 8` next to a 400 KB corpus reads as "the whole corpus"; it is the
    first 4096 tokens. The window has to be in the report for the same reason the
    corpus hash is."""
    got = parse_ppl(REAL_LOG)
    assert got["chunks_scored"] == 8
    assert got["ctx"] == 512
    assert got["tokens_scored"] == 4096


def test_parse_ppl_window_is_absent_when_the_run_died_before_it():
    got = parse_ppl("0.00.930.529 W llama_model_loader: tensor overrides\n")
    assert got["tokens_scored"] is None
    assert window_note(got, 400_000) is None      # no window -> no claim either way


def test_window_note_flags_a_partial_corpus():
    run = parse_ppl(REAL_LOG)
    note = window_note(run, 397_855)
    assert note is not None
    assert "4096 tokens" in note and "FIRST" in note


def test_window_note_is_silent_when_the_whole_corpus_was_scored():
    """A corpus that fits inside the window is measured in full, and must not be
    reported as partial -- an always-on warning is one nobody reads."""
    run = parse_ppl(REAL_LOG)
    assert window_note(run, 12_000) is None


# ---------------------------------------------------------------------------
# corpus: the comparison is only valid on identical tokens
# ---------------------------------------------------------------------------

def test_corpus_is_deterministic_and_repo_local(tmp_path):
    a = build_corpus(str(tmp_path / "a.txt"))
    b = build_corpus(str(tmp_path / "b.txt"))
    assert a["sha256"] == b["sha256"]
    assert a["bytes"] > 1000
    assert a["files"] == b["files"]
    # Prose first, then code, so two runs get the same token stream.
    assert a["files"][0] == "README.md"
    assert any(f.startswith("experiments/") for f in a["files"])
    assert any(f.startswith("src/moeme/") for f in a["files"])


def test_corpus_marks_each_source(tmp_path):
    """Source boundaries are written into the text, so a reader can attribute
    which part of the corpus produced a chunk."""
    c = build_corpus(str(tmp_path / "c.txt"))
    body = open(c["path"], "rb").read().decode("utf-8", "replace")
    assert "===== README.md =====" in body


# ---------------------------------------------------------------------------
# quant name from the GGUF header
# ---------------------------------------------------------------------------

def _little_gguf(path, file_type, *, version=3, tokens=0):
    """A GGUF header whose file_type key sits behind an optional token array.

    Real artifacts carry the tokenizer's whole vocabulary as a string array, and
    `general.file_type` comes after it, so this is the shape the reader has to
    survive rather than the easy one-key file it was first written against.
    """
    counts = "<Q" if version >= 3 else "<I"
    with open(path, "wb") as fh:
        fh.write(b"GGUF")
        fh.write(struct.pack("<I", version))
        fh.write(struct.pack(counts, 0))                    # n_tensors
        fh.write(struct.pack(counts, 1 + (1 if tokens else 0)))
        if tokens:
            key = b"tokenizer.ggml.tokens"
            fh.write(struct.pack("<Q", len(key)) + key)
            fh.write(struct.pack("<I", 9))                  # GGUF_TYPE_ARRAY
            fh.write(struct.pack("<I", 8))                  # ... of STRING
            fh.write(struct.pack("<Q", tokens))
            for i in range(tokens):
                tok = f"tok{i}".encode()
                fh.write(struct.pack("<Q", len(tok)) + tok)
        key = b"general.file_type"
        fh.write(struct.pack("<Q", len(key)) + key)
        fh.write(struct.pack("<I", 4))                      # GGUF_TYPE_UINT32
        fh.write(struct.pack("<I", file_type))


def test_fmt_bpw_names_the_quantisation(tmp_path):
    p = str(tmp_path / "m.gguf")
    _little_gguf(p, 15)
    assert fmt_bpw(p) == "Q4_K_M"
    _little_gguf(p, 10)
    assert fmt_bpw(p) == "Q2_K"


def test_fmt_bpw_reads_a_file_type_behind_the_token_array(tmp_path):
    """Regression. This is the bug that shipped and then showed up as `"quant": "?"`
    in the very report the field exists to fill: the reader consumed arrays by
    their printed width, lost alignment on the tokenizer array, and answered "?"
    for both of ~/models' 35B GGUFs without saying anything was wrong."""
    p = str(tmp_path / "m.gguf")
    _little_gguf(p, 10, tokens=300)
    assert fmt_bpw(p) == "Q2_K"
    _little_gguf(p, 15, tokens=300)
    assert fmt_bpw(p) == "Q4_K_M"


def test_fmt_bpw_reads_a_v2_header_with_uint32_counts(tmp_path):
    """GGUF v1/v2 store the two counts as uint32; v3 widened them. Reading a
    v2 header with v3 widths lands mid-key and reports "?" as if the file had no
    file_type."""
    p = str(tmp_path / "old.gguf")
    _little_gguf(p, 18, version=2)
    assert fmt_bpw(p) == "Q6_K"


def test_fmt_bpw_reports_unknown_rather_than_guessing_from_the_name(tmp_path):
    """A file that does not state its quantisation gets "?", never a name read off
    the filename -- the report is quoted as a measurement."""
    p = str(tmp_path / "claims-to-be-q2_k.gguf")
    _little_gguf(p, 9999)
    assert fmt_bpw(p) == "?"


def test_fmt_bpw_refuses_to_guess_on_a_broken_file(tmp_path):
    p = str(tmp_path / "nope.gguf")
    with open(p, "wb") as fh:
        fh.write(b"not a gguf at all")
    assert fmt_bpw(p) == "?"
    # A truncated header is the other way a "simple" reader lies: it reads past
    # the end and reports whatever it found in the buffer.
    truncated = str(tmp_path / "cut.gguf")
    _little_gguf(truncated, 15)
    raw = open(truncated, "rb").read()
    open(truncated, "wb").write(raw[:24])
    assert fmt_bpw(truncated) == "?"


def test_gguf_metadata_keeps_alignment_while_selecting_keys(tmp_path):
    p = str(tmp_path / "m.gguf")
    _little_gguf(p, 15, tokens=40)
    meta = gguf_metadata(p, keys={"general.file_type"})
    assert meta == {"general.file_type": 15}
    assert "tokenizer.ggml.tokens" not in meta
    everything = gguf_metadata(p)
    assert everything["general.file_type"] == 15
    assert everything["tokenizer.ggml.tokens"][:2] == ["tok0", "tok1"]


# ---------------------------------------------------------------------------
# self-test entry point
# ---------------------------------------------------------------------------

def test_self_test_passes_offline():
    assert quant_perplexity.self_test() == 0
