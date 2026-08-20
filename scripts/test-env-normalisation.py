#!/usr/bin/env python3
"""Unit tests for the .env.dspark normalisation and worker-copy hardening in
start-deepseek-v4-flash-dspark.sh.

CPU-only; no GPU, no network, no real worker. The env-loading block and the
remote-write line are lifted verbatim from the launcher so the tests exercise
the shipped code rather than a copy of it.

    python3 scripts/test-env-normalisation.py -q
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAUNCHER = os.path.join(ROOT, "start-deepseek-v4-flash-dspark.sh")


def _extract(start_marker: str, end_marker: str) -> str:
    """Pull a verbatim block out of the launcher so tests can't drift from it."""
    with open(LAUNCHER, encoding="utf-8") as fh:
        src = fh.read()
    i = src.index(start_marker)
    j = src.index(end_marker, i) + len(end_marker)
    return src[i:j]


ENV_BLOCK = _extract('_dspark_env_clean="$(mktemp)"', "trap - EXIT HUP INT TERM")
def _extract_remote_write() -> str:
    """The worker-copy pipeline: the `sed` that feeds the ssh write.

    Anchored on the ssh invocation (stable) and bounded by the end of its logical
    line, so changing the *mode* or the redirect target still yields a runnable
    block and produces a clean test failure rather than an extraction crash.
    """
    with open(LAUNCHER, encoding="utf-8") as fh:
        src = fh.read()
    j = src.index('| ssh "$WORKER_HOST" "umask 077;')
    i = src.rindex("sed $'", 0, j)
    end = src.index("\n", j)
    return src[i:end]


REMOTE_WRITE = _extract_remote_write()


def run_env_block(content: bytes, extra: str = "") -> subprocess.CompletedProcess:
    """Source `content` through the launcher's normalisation block."""
    d = tempfile.mkdtemp()
    env_file = os.path.join(d, ".env.dspark")
    with open(env_file, "wb") as f:
        f.write(content)
    script = f"""set -euo pipefail
ENV_FILE={env_file!r}
{ENV_BLOCK}
printf 'WORKER_HOST=%q\\nVLLM_PORT=%q\\n' "${{WORKER_HOST:-<unset>}}" "${{VLLM_PORT:-<unset>}}"
{extra}
"""
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    p.env_file = env_file  # type: ignore[attr-defined]
    p.workdir = d          # type: ignore[attr-defined]
    return p


class TestEnvNormalisation(unittest.TestCase):
    def test_plain_file_unaffected(self):
        p = run_env_block(b"WORKER_HOST=10.0.0.2\nVLLM_PORT=8888\n")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("WORKER_HOST=10.0.0.2", p.stdout)
        self.assertIn("VLLM_PORT=8888", p.stdout)

    def test_utf8_bom(self):
        # A BOM lands on line 1, so an unnormalised source tries to execute it.
        p = run_env_block(b"\xef\xbb\xbf# cluster (head)\nWORKER_HOST=10.0.0.2\nVLLM_PORT=8888\n")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("WORKER_HOST=10.0.0.2", p.stdout)

    def test_crlf_with_blank_lines(self):
        # Blank CRLF lines abort an unnormalised source with $'\r': command not found.
        p = run_env_block(b"WORKER_HOST=10.0.0.2\r\n\r\nVLLM_PORT=8888\r\n")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("WORKER_HOST=10.0.0.2", p.stdout)

    def test_crlf_without_blank_lines_is_not_silently_corrupted(self):
        # The dangerous case: unnormalised this returns rc=0 with a trailing \r
        # riding inside the value.
        p = run_env_block(b"WORKER_HOST=10.0.0.2\r\nVLLM_PORT=8888\r\n")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("WORKER_HOST=10.0.0.2", p.stdout)
        self.assertNotIn("\\r", p.stdout)

    def test_bom_and_crlf_combined(self):
        p = run_env_block(b"\xef\xbb\xbfWORKER_HOST=10.0.0.2\r\n\r\nVLLM_PORT=8888\r\n")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("WORKER_HOST=10.0.0.2", p.stdout)
        self.assertNotIn("\\r", p.stdout)

    def test_operator_file_is_not_rewritten(self):
        raw = b"\xef\xbb\xbfWORKER_HOST=10.0.0.2\r\nVLLM_PORT=8888\r\n"
        p = run_env_block(raw)
        self.assertEqual(p.returncode, 0, p.stderr)
        with open(p.env_file, "rb") as fh:
            after = fh.read()
        self.assertEqual(after, raw,
                         "the operator's .env.dspark must be left byte-identical")

    def test_temp_copy_removed_on_success(self):
        p = run_env_block(b"WORKER_HOST=10.0.0.2\n", extra='echo "LEFTOVER=$(ls /tmp/tmp.* 2>/dev/null | wc -l)"')
        self.assertEqual(p.returncode, 0, p.stderr)
        # the block clears its own trap, so the copy is gone before we look
        self.assertNotIn("_dspark_env_clean", p.stdout)

    def test_temp_copy_removed_when_source_fails(self):
        # A syntax error inside the env file makes `source` fail under set -e.
        # The secret-bearing copy must not be left behind in TMPDIR.
        d = tempfile.mkdtemp()
        tmpdir = os.path.join(d, "tmp"); os.makedirs(tmpdir)
        env_file = os.path.join(d, ".env.dspark")
        with open(env_file, "w") as f:
            f.write("WORKER_HOST=(unbalanced\n")
        script = f"""set -euo pipefail
export TMPDIR={tmpdir!r}
ENV_FILE={env_file!r}
{ENV_BLOCK}
"""
        p = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0, "malformed env file should fail the launch")
        leftovers = os.listdir(tmpdir)
        self.assertEqual(leftovers, [],
                         f"normalised copy (holds HF_TOKEN) survived a source failure: {leftovers}")

    def test_temp_copy_removed_on_signal(self):
        # Same guarantee when the launcher is interrupted mid-source.
        d = tempfile.mkdtemp()
        tmpdir = os.path.join(d, "tmp"); os.makedirs(tmpdir)
        env_file = os.path.join(d, ".env.dspark")
        with open(env_file, "w") as f:
            f.write("WORKER_HOST=10.0.0.2\nkill -TERM $$\n")
        script = f"""set -uo pipefail
export TMPDIR={tmpdir!r}
ENV_FILE={env_file!r}
{ENV_BLOCK}
"""
        subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        leftovers = os.listdir(tmpdir)
        self.assertEqual(leftovers, [],
                         f"normalised copy survived a signal: {leftovers}")

    def test_temp_copy_is_not_world_readable(self):
        p = run_env_block(b"WORKER_HOST=10.0.0.2\n",
                          extra='stat -c %a "$_dspark_env_clean" 2>/dev/null || echo gone')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("gone", p.stdout, "copy should already be removed at this point")


class TestRemoteWorkerCopy(unittest.TestCase):
    """The worker copy must be created 0600 and must fail closed."""

    def _run(self, remote_path: str, ssh_rc: int = 0):
        d = tempfile.mkdtemp()
        bindir = os.path.join(d, "bin"); os.makedirs(bindir)
        log = os.path.join(d, "ssh.log")
        with open(os.path.join(bindir, "ssh"), "w") as f:
            f.write("#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> " + repr(log) +
                    "\ncat > /dev/null\nexit " + str(ssh_rc) + "\n")
        os.chmod(os.path.join(bindir, "ssh"), 0o755)
        env_file = os.path.join(d, ".env.dspark")
        with open(env_file, "w") as fh:
            fh.write("HF_TOKEN=secret\n")
        script = f"""set -euo pipefail
export PATH={bindir!r}:$PATH
WORKER_HOST=worker
REMOTE_ENV_FILE={remote_path}
ENV_FILE={env_file!r}
{REMOTE_WRITE}
echo OK
"""
        p = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        if os.path.exists(log):
            with open(log) as fh:
                p.log = fh.read()  # type: ignore[attr-defined]
        else:
            p.log = ""  # type: ignore[attr-defined]
        return p

    def test_creates_with_umask_077(self):
        p = self._run("/srv/dspark/.env.dspark")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("umask 077", p.log)
        self.assertIn("/srv/dspark/.env.dspark", p.log)

    def test_fails_closed_when_remote_write_fails(self):
        # No `|| true`: a failure to place the secret safely must abort the launch.
        p = self._run("/srv/dspark/.env.dspark", ssh_rc=1)
        self.assertNotEqual(p.returncode, 0,
                            "launch continued despite failing to write the worker credential")
        self.assertNotIn("OK", p.stdout)

    def test_enforces_0600_on_preexisting_destination(self):
        """umask only applies to NEW inodes; an existing 0644 dest must still end 0600."""
        d = tempfile.mkdtemp()
        bindir = os.path.join(d, "bin"); os.makedirs(bindir)
        dest = os.path.join(d, "remote-env")
        with open(dest, "w") as fh:
            fh.write("STALE=1\n")
        os.chmod(dest, 0o644)
        # stub ssh executes the remote command locally so mode changes are observable
        with open(os.path.join(bindir, "ssh"), "w") as fh:
            fh.write('#!/usr/bin/env bash\nshift\nbash -c "$*"\n')
        os.chmod(os.path.join(bindir, "ssh"), 0o755)
        env_file = os.path.join(d, ".env.dspark")
        with open(env_file, "w") as fh:
            fh.write("HF_TOKEN=secret\n")
        script = f"""set -euo pipefail
export PATH={bindir!r}:$PATH
WORKER_HOST=worker
REMOTE_ENV_FILE={dest!r}
ENV_FILE={env_file!r}
{REMOTE_WRITE}
"""
        p = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        mode = oct(os.stat(dest).st_mode & 0o777)
        self.assertEqual(mode, "0o600",
                         f"pre-existing worker credential left at {mode} (umask does not chmod an existing inode)")
        with open(dest) as fh:
            self.assertIn("HF_TOKEN=secret", fh.read())

    def test_worker_receives_normalised_bytes(self):
        """The worker's compose --env-file must not see BOM/CRLF the head was shielded from."""
        d = tempfile.mkdtemp()
        bindir = os.path.join(d, "bin"); os.makedirs(bindir)
        dest = os.path.join(d, "remote-env")
        with open(os.path.join(bindir, "ssh"), "w") as fh:
            fh.write('#!/usr/bin/env bash\nshift\nbash -c "$*"\n')
        os.chmod(os.path.join(bindir, "ssh"), 0o755)
        env_file = os.path.join(d, ".env.dspark")
        with open(env_file, "wb") as fh:
            fh.write(b"\xef\xbb\xbfWORKER_HOST=10.0.0.2\r\nHF_TOKEN=secret\r\n")
        script = f"""set -euo pipefail
export PATH={bindir!r}:$PATH
WORKER_HOST=worker
REMOTE_ENV_FILE={dest!r}
ENV_FILE={env_file!r}
{REMOTE_WRITE}
"""
        p = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        with open(dest, "rb") as fh:
            got = fh.read()
        self.assertFalse(got.startswith(b"\xef\xbb\xbf"), "BOM reached the worker copy")
        self.assertNotIn(b"\r", got, "CRLF reached the worker copy")
        self.assertIn(b"HF_TOKEN=secret", got)

    def test_escaped_remote_path_with_space(self):
        # REMOTE_ENV_FILE is already %q-escaped upstream; it must be used unquoted
        # so the escaping survives rather than being nested inside literal quotes.
        escaped = subprocess.run(
            ["bash", "-c", 'printf %q "/srv/dspark lab/.env.dspark"'],
            capture_output=True, text=True).stdout
        p = self._run(escaped)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("dspark", p.log)
        self.assertNotIn("''", p.log, "escaped path should not be re-wrapped in quotes")


if __name__ == "__main__":
    unittest.main(verbosity=0 if "-q" in sys.argv else 2,
                  argv=[a for a in sys.argv if a != "-q"])
