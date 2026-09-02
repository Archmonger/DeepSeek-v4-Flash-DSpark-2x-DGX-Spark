#!/usr/bin/env python3
"""Apply the Markov-replicate hotfix to the real image qwen3_dspark.py."""
from __future__ import annotations

import os
import py_compile
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOTFIX = ROOT / "patches" / "hotfix-dsv4-replicate-markov-head.py"
REAL = (
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/"
    "qwen3_dspark.py"
)
IMAGE = "ghcr.io/anemll/dspark-vllm-gx10:0.1.1"


def extract_real() -> Path:
    fd, path = tempfile.mkstemp(suffix=".py", text=True)
    os.close(fd)
    dest = Path(path)
    subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "cat", IMAGE, REAL],
        check=True,
        stdout=dest.open("w"),
    )
    return dest


def apply_to(path: Path) -> None:
    txt = HOTFIX.read_text()
    marker = (
        "P = Path(\n"
        f'    "{REAL}"\n'
        ")"
    )
    if marker not in txt:
        raise AssertionError("hotfix Path() constructor drifted")
    txt = txt.replace(marker, f"P = Path({str(path)!r})", 1)
    ns: dict = {}
    exec(compile(txt, str(HOTFIX), "exec"), ns)


def main() -> int:
    tmpd = Path(tempfile.mkdtemp())
    try:
        real = extract_real()
        base = tmpd / "qwen3_dspark.py"
        shutil.copy(real, base)
        src = base.read_text()
        assert "class DSparkMarkovHead" in src
        assert "VocabParallelEmbedding(" in src
        apply_to(base)
        patched = base.read_text()
        assert "# [dspark-replicate-markov]" in patched
        assert "nn.Embedding(vocab_size, markov_rank)" in patched
        assert "ReplicatedLinear(" in patched
        assert "logits_processor(self.markov_w2" not in patched
        py_compile.compile(str(base), doraise=True)
        try:
            apply_to(base)
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
        assert base.read_text() == patched
        print("PASS: replicate-markov hotfix applies, compiles, is idempotent")
        return 0
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
