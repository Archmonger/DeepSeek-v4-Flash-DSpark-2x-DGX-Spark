#!/usr/bin/env python3
"""Unit tests for numeric-knob validation in the DSpark recipe.

MAX_NUM_SEQS / MTP_NUM_TOKENS / MAX_NUM_BATCHED_TOKENS are interpolated into bash
arithmetic (and forwarded into the container's own $(( )) via compose). Without
validation, a malformed value either aborts with a cryptic arithmetic error, is
accepted as a false success by the validator, or is silently misread as octal
(010 -> 8). This exercises the validation block lifted verbatim from the launcher.

    python3 scripts/test-numeric-knob-validation.py -q

CPU-only; no GPU, no container, no network.
"""
import os
import re
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAUNCHER = os.path.join(ROOT, "start-deepseek-v4-flash-dspark.sh")


def _extract_block() -> str:
    """Lift the validation for-loop verbatim so the test can't drift from shipped code."""
    with open(LAUNCHER, encoding="utf-8") as _fh:
        src = _fh.read()
    i = src.index("for _dspark_num in MAX_NUM_SEQS")
    j = src.index("unset _dspark_num _dspark_val", i) + len("unset _dspark_num _dspark_val")
    return src[i:j]


BLOCK = _extract_block()


def run(var: str, val: str):
    """Run the validation block with <var>=<val>, then echo the normalized value."""
    script = f"""set -euo pipefail
_dspark_env_clean=
{var}={val!r}
export {var}
{BLOCK}
printf 'OK %s=%s cap=%s\\n' "{var}" "${{{var}:-<unset>}}" \
  "$(( ${{MAX_NUM_SEQS:-6}} * (${{MTP_NUM_TOKENS:-5}} + 1) ))"
"""
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)


class TestNumericValidation(unittest.TestCase):
    def test_valid_passes(self):
        rc, out = run("MAX_NUM_SEQS", "6")
        self.assertEqual(rc, 0, out)
        self.assertIn("MAX_NUM_SEQS=6", out)
        self.assertIn("cap=36", out)

    def test_decimal_rejected(self):
        rc, out = run("MAX_NUM_SEQS", "6.5")
        self.assertEqual(rc, 2, out)
        self.assertIn("MAX_NUM_SEQS must be a non-negative integer", out)

    def test_alnum_rejected(self):
        rc, out = run("MAX_NUM_SEQS", "6x")
        self.assertEqual(rc, 2, out)
        self.assertIn("must be a non-negative integer", out)

    def test_bareword_rejected(self):
        rc, out = run("MAX_NUM_SEQS", "eight")
        self.assertEqual(rc, 2, out)
        self.assertIn("must be a non-negative integer", out)

    def test_crlf_rejected(self):
        rc, out = run("MAX_NUM_SEQS", "6\r")
        self.assertEqual(rc, 2, out)
        self.assertIn("must be a non-negative integer", out)

    def test_leading_zero_normalized_not_octal(self):
        # The core octal fix: 010 must mean decimal 10 (cap 60), never octal 8 (cap 48).
        rc, out = run("MAX_NUM_SEQS", "010")
        self.assertEqual(rc, 0, out)
        self.assertIn("MAX_NUM_SEQS=10", out)
        self.assertIn("cap=60", out)

    def test_leading_zero_eight(self):
        rc, out = run("MAX_NUM_SEQS", "08")
        self.assertEqual(rc, 0, out)
        self.assertIn("MAX_NUM_SEQS=8", out)
        self.assertIn("cap=48", out)

    def test_mtp_bareword_rejected(self):
        rc, out = run("MTP_NUM_TOKENS", "five")
        self.assertEqual(rc, 2, out)
        self.assertIn("MTP_NUM_TOKENS must be a non-negative integer", out)

    def test_batched_tokens_rejected(self):
        rc, out = run("MAX_NUM_BATCHED_TOKENS", "bad")
        self.assertEqual(rc, 2, out)
        self.assertIn("MAX_NUM_BATCHED_TOKENS must be a non-negative integer", out)

    def test_empty_uses_default(self):
        rc, out = run("MAX_NUM_SEQS", "")
        self.assertEqual(rc, 0, out)
        self.assertIn("cap=36", out)  # :- default of 6 applies


if __name__ == "__main__":
    unittest.main(verbosity=0 if "-q" in sys.argv else 2,
                  argv=[a for a in sys.argv if a != "-q"])
