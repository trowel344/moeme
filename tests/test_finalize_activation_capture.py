import struct
from pathlib import Path

import numpy as np

from moeme.activations import MAGIC, VERSION
from scripts.finalize_activation_capture import finalize_capture


def test_finalize_stream_validates_and_atomically_publishes_capture(tmp_path: Path) -> None:
    partial = tmp_path / ".capture.partial"
    output = tmp_path / "capture"
    corpus = tmp_path / "corpus.txt"
    partial.mkdir()
    corpus.write_text("recovery corpus")
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    with (partial / "layer-63.f32").open("wb") as handle:
        handle.write(struct.pack("<III", MAGIC, VERSION, 4))
        handle.write(struct.pack("<I", 3))
        handle.write(values.tobytes())
    manifest = finalize_capture(partial, output, corpus, [63], 3, 0, True)
    assert not partial.exists()
    assert manifest["recovered_from_completed_temporary"] is True
    assert manifest["layers"]["63"]["tokens"] == 3
    assert (output / "manifest.json").is_file()
