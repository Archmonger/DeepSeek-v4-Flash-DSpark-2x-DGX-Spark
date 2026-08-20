#!/usr/bin/env python3
"""CPU gates for DSPARK_API_KEYS multi-key auth (behavioral).

These tests execute the REAL auth code — extracted from docker-compose.dspark.yml
(the single-line entrypoint block) and from the three probe scripts (the
`# DSPARK_API_KEYS auth ...` marker blocks) — through a shell, and assert on
observable behavior rather than source text:

- unset / empty / whitespace-only DSPARK_API_KEYS adds no auth anywhere;
- a parsed value becomes EXACTLY ONE `--api-key` flag carrying every key
  (order preserved, separators collapsed, duplicates allowed). Repeating the
  flag overwrites instead of appends in vLLM (nargs with last-wins), so a
  per-key loop would silently leave only the last key valid;
- literal glob characters survive as literal argv/header tokens (no pathname
  expansion, no word-splitting inside an element);
- a token starting with `-` is rejected with exit 2 in all four contexts;
- VLLM_API_KEY and DSPARK_API_KEYS both meaningful => exit 2 naming BOTH
  variables, in the compose entrypoint and in all three probe scripts. The
  server must never silently choose one (vLLM's CLI would win) and the probes
  must never guess which variable the server honoured;
- VLLM_API_KEY alone keeps the legacy single-key path unchanged;
- the three probe blocks are byte-identical (consistent parsing);
- Compose interpolates the key variables container-side (`$$`), so key text is
  never host-interpolated into shell source by Compose.

Negative controls run the gates against deliberately regressed variants
(`$` instead of `$$`, per-key repeated flags) and require the gate to FAIL,
proving it would catch that regression.

The compose `command` is one YAML folded scalar whose base-indent lines are
joined with spaces; the auth block is a single physical line at that indent,
so its text survives folding verbatim. The `exec vllm serve ...` tail is all
base-indent lines, so joining them with spaces reproduces the folded scalar for
that region. `$$` -> `$` is applied exactly the way Compose passes a literal
`$` through; host-side single-`$` refs (e.g. ${DSPARK_MODEL:-...}) are left
for bash's own ${VAR:-default} expansion, which matches runtime.

No GPU, no serve, stdlib only.
"""

import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "docker-compose.dspark.yml"
ENV_EXAMPLE = ROOT / ".env.dspark.example"
ENVS_DOC = ROOT / "docs" / "ENVS.md"
PROBES = (
    ROOT / "start-deepseek-v4-flash-dspark.sh",
    ROOT / "smoke-deepseek-v4-flash-dspark.sh",
    ROOT / "status-deepseek-v4-flash-dspark.sh",
)

BLOCK_START = '_dspark_keys_set=0; case "$${DSPARK_API_KEYS:-}" in'
BLOCK_END = 'API_KEY_ARGS=(--api-key "$${_dspark_keys[@]}"); fi;'
PROBE_BEGIN = "# DSPARK_API_KEYS auth (begin)"
PROBE_END = "# DSPARK_API_KEYS auth (end)"


# --------------------------------------------------------------------------
# Extraction: get the real code out of the real files.
# --------------------------------------------------------------------------


def entrypoint_auth_block() -> str:
    """The compose entrypoint's auth line, still carrying Compose `$$` escapes."""
    for line in COMPOSE.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith(BLOCK_START):
            stripped = line.strip()
            if not stripped.endswith("fi;"):
                raise AssertionError(
                    "entrypoint auth block must remain ONE physical compose line"
                )
            return stripped
    raise AssertionError("entrypoint auth block not found in docker-compose.dspark.yml")


def entrypoint_exec_tail() -> str:
    """The `exec vllm serve ...` tail as the container sees it ($$ already `$`)."""
    text = COMPOSE.read_text(encoding="utf-8")
    start = text.index("exec /usr/local/bin/vllm serve")
    line_start = text.rindex("\n", 0, start) + 1
    tail_lines = []
    for line in text[line_start:].splitlines():
        if not line.strip():
            break
        if len(line) - len(line.lstrip(" ")) < 8:
            break
        tail_lines.append(line.strip())
    if not tail_lines:
        raise AssertionError("exec vllm serve tail not found in compose")
    return " ".join(tail_lines).replace("$$", "$")


def probe_auth_block(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    begin = text.index(PROBE_BEGIN) + len(PROBE_BEGIN)
    end = text.index(PROBE_END, begin)
    return text[begin:end]


# --------------------------------------------------------------------------
# Shell helpers.
# --------------------------------------------------------------------------


def run_bash(body: str, env_kwargs, cwd=None):
    env = dict(os.environ)
    for key, value in env_kwargs.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    env.setdefault("VLLM_API_KEY", "")
    return subprocess.run(
        ["bash", "-c", body], capture_output=True, text=True, env=env, cwd=cwd
    )


def run_entrypoint(env_kwargs, cwd=None):
    """Run the real entrypoint auth block and print the resulting API_KEY_ARGS."""
    block = entrypoint_auth_block().replace("$$", "$")
    body = (
        block + "\n"
        + "printf 'COUNT=%s\\n' \"${#API_KEY_ARGS[@]}\";\n"
        + 'printf "<%s>\\n" "${API_KEY_ARGS[@]}"\n'
    )
    return run_bash(body, env_kwargs, cwd)


def run_probe(path: Path, env_kwargs):
    """Run a probe's real auth block and print the resulting AUTH_HEADER_ARGS."""
    body = (
        probe_auth_block(path) + "\n"
        + "printf 'COUNT=%s\\n' \"${#AUTH_HEADER_ARGS[@]}\";\n"
        + 'printf "<%s>\\n" "${AUTH_HEADER_ARGS[@]}"\n'
    )
    return run_bash(body, env_kwargs)


def argv_of(env_kwargs):
    """Run the real block + exec tail and return the argv handed to vllm serve."""
    block = entrypoint_auth_block().replace("$$", "$")
    tail = entrypoint_exec_tail().replace(
        "exec /usr/local/bin/vllm serve",
        "python3 -c 'import sys; [print(\"A[\"+a+\"]\") for a in sys.argv[1:]]' ",
    )
    r = run_bash(block + "\n" + tail, env_kwargs)
    if r.returncode != 0:
        raise AssertionError(f"argv dump failed:\n{r.stderr}")
    return [ln[2:-1] for ln in r.stdout.splitlines()
            if ln.startswith("A[") and ln.endswith("]")]


_SINGLE_DOLLAR = re.compile(
    r'(?<!\$)\$\{(DSPARK_API_KEYS|VLLM_API_KEY|_dspark_keys_set'
    r'|_dspark_keys\[@\]|_dspark_keys|_dspark_key)\}'
)


def single_dollar_refs(text: str):
    """Compose host-side interpolation refs (a `$` that is not `$$`)."""
    return _SINGLE_DOLLAR.findall(text)


# --------------------------------------------------------------------------
# Compose entrypoint: argv construction.
# --------------------------------------------------------------------------


class EntrypointArgv(unittest.TestCase):
    def test_unset_empty_whitespace_only_add_no_args(self):
        for env in ({}, {"DSPARK_API_KEYS": ""}, {"DSPARK_API_KEYS": "   \t "}):
            r = run_entrypoint(env)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(r.stdout.startswith("COUNT=0"), (env, r.stdout))

    def test_single_key_is_one_flag(self):
        r = run_entrypoint({"DSPARK_API_KEYS": "k1"})
        self.assertEqual(r.stdout, "COUNT=2\n<--api-key>\n<k1>\n", r.stderr)

    def test_many_keys_one_flag_exactly_order_kept(self):
        r = run_entrypoint({"DSPARK_API_KEYS": "k1 k2 k3"})
        elems = r.stdout.splitlines()
        self.assertEqual(elems[0], "COUNT=4")
        self.assertEqual(elems[1:], ["<--api-key>", "<k1>", "<k2>", "<k3>"])

    def test_separators_collapsed_order_kept(self):
        for value in ("k1  k2   k3", "  k1 k2\tk3  ", "k1   \t  k2 k3"):
            r = run_entrypoint({"DSPARK_API_KEYS": value})
            elems = r.stdout.splitlines()
            self.assertEqual(elems[1:], ["<--api-key>", "<k1>", "<k2>", "<k3>"], value)

    def test_duplicates_allowed(self):
        r = run_entrypoint({"DSPARK_API_KEYS": "k1 k1 k1"})
        elems = r.stdout.splitlines()
        self.assertEqual(elems[1:], ["<--api-key>", "<k1>", "<k1>", "<k1>"])

    def test_literal_glob_chars_not_expanded(self):
        with tempfile.TemporaryDirectory() as td:
            Path(td, "k2tail").write_text("x")
            r = run_entrypoint({"DSPARK_API_KEYS": "k1 k2*"}, cwd=td)
        elems = r.stdout.splitlines()
        self.assertEqual(elems[1:], ["<--api-key>", "<k1>", "<k2*>"])

    def test_dash_leading_token_rejected_with_token_named(self):
        for value in ("-bad", "k1 -bad", "k1  -bad  k3"):
            r = run_entrypoint({"DSPARK_API_KEYS": value})
            self.assertEqual(r.returncode, 2, value)
            self.assertIn("-bad", r.stderr, value)

    def test_both_vars_named_and_exit_2(self):
        r = run_entrypoint({"VLLM_API_KEY": "vk", "DSPARK_API_KEYS": "k1 k2"})
        self.assertEqual(r.returncode, 2)
        self.assertIn("VLLM_API_KEY", r.stderr)
        self.assertIn("DSPARK_API_KEYS", r.stderr)

    def test_vllm_only_adds_no_flag(self):
        # VLLM_API_KEY is vLLM's own env var (served natively); the entrypoint
        # must not duplicate it into --api-key, where the CLI would silently
        # override the env value.
        r = run_entrypoint({"VLLM_API_KEY": "vk"})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(r.stdout.startswith("COUNT=0"), r.stdout)

    def test_full_argv_exactly_one_api_key_flag(self):
        argv = argv_of({"DSPARK_API_KEYS": "sk-a sk-b sk-c"})
        self.assertEqual(argv.count("--api-key"), 1, argv)
        i = argv.index("--api-key")
        self.assertEqual(argv[i + 1:i + 4], ["sk-a", "sk-b", "sk-c"])

    def test_full_argv_no_api_key_when_unset(self):
        argv = argv_of({})
        self.assertNotIn("--api-key", argv)


class EntrypointInterpolation(unittest.TestCase):
    """Compose `$$` vs `$` on the key variables (container-side expansion)."""

    def test_entrypoint_uses_double_dollar(self):
        block = entrypoint_auth_block()
        self.assertIn("$${DSPARK_API_KEYS", block)
        self.assertIn("$${VLLM_API_KEY", block)
        self.assertEqual(single_dollar_refs(block), [],
                         "host-side interpolation refs in entrypoint block")

    def test_single_dollar_regression_is_caught(self):
        regressed = entrypoint_auth_block().replace("$${", "${")
        self.assertTrue(single_dollar_refs(regressed),
                        "regression gate did not detect `$` in place of `$$`")

    def test_repeated_flag_regression_is_caught(self):
        # Happy path: exactly one --api-key for N keys.
        happy = run_entrypoint({"DSPARK_API_KEYS": "k1 k2"})
        self.assertEqual(happy.stdout.count("--api-key"), 1, happy.stdout)
        # Regress: per-key loop (would leave only the last key valid in vLLM).
        regressed = entrypoint_auth_block().replace(
            'API_KEY_ARGS=(--api-key "$${_dspark_keys[@]}"); fi;',
            'for _rk in "$${_dspark_keys[@]}"; do API_KEY_ARGS+=(--api-key "$${_rk}"); done; fi;',
        ).replace("$$", "$")
        rr = run_bash(
            regressed + "\n" + 'printf "%s\\n" "${API_KEY_ARGS[@]}"',
            {"DSPARK_API_KEYS": "k1 k2"},
        )
        self.assertGreater(rr.stdout.count("--api-key"), 1,
                           "regression gate missed repeated --api-key flags")


# --------------------------------------------------------------------------
# Probe scripts: header selection and conflict handling.
# --------------------------------------------------------------------------


class ProbeAuth(unittest.TestCase):
    def test_probes_consistent_identical_blocks(self):
        texts = {probe_auth_block(p) for p in PROBES}
        self.assertEqual(len(texts), 1, "probe auth blocks must stay identical")

    def test_probes_no_auth_when_unset_empty_whitespace(self):
        for path in PROBES:
            for env in ({}, {"DSPARK_API_KEYS": ""}, {"DSPARK_API_KEYS": "  \t "}):
                r = run_probe(path, env)
                self.assertEqual(r.returncode, 0, (path.name, env))
                self.assertTrue(r.stdout.startswith("COUNT=0"), (path.name, env))

    def test_probes_use_first_parsed_key(self):
        for path in PROBES:
            for value, want in (("k1", "k1"), (" k2  k1 ", "k2"),
                                ("k3 k2 k1", "k3"), ("\tk4 k1", "k4")):
                r = run_probe(path, {"DSPARK_API_KEYS": value})
                self.assertEqual(r.returncode, 0, (path.name, value))
                self.assertIn(f"<Authorization: Bearer {want}>", r.stdout,
                              (path.name, value))

    def test_probes_literal_glob_first_key(self):
        for path in PROBES:
            r = run_probe(path, {"DSPARK_API_KEYS": "g* k1"})
            self.assertIn("<Authorization: Bearer g*>", r.stdout, path.name)

    def test_probes_reject_dash_leading_token(self):
        for path in PROBES:
            r = run_probe(path, {"DSPARK_API_KEYS": "k1 -bad"})
            self.assertEqual(r.returncode, 2, path.name)
            self.assertIn("-bad", r.stderr, path.name)

    def test_probes_exit_2_naming_both_vars(self):
        for path in PROBES:
            r = run_probe(path, {"VLLM_API_KEY": "vk", "DSPARK_API_KEYS": "k1"})
            self.assertEqual(r.returncode, 2, path.name)
            self.assertIn("VLLM_API_KEY", r.stderr, path.name)
            self.assertIn("DSPARK_API_KEYS", r.stderr, path.name)

    def test_probes_vllm_only_unchanged(self):
        for path in PROBES:
            r = run_probe(path, {"VLLM_API_KEY": "vk"})
            self.assertEqual(r.returncode, 0, path.name)
            self.assertEqual(r.stdout,
                             "COUNT=2\n<-H>\n<Authorization: Bearer vk>\n",
                             path.name)


# --------------------------------------------------------------------------
# Docs + hygiene.
# --------------------------------------------------------------------------


class Documented(unittest.TestCase):
    def test_env_example_documents_contract(self):
        text = ENV_EXAMPLE.read_text(encoding="utf-8")
        self.assertIn("DSPARK_API_KEYS", text)
        self.assertIn("VLLM_API_KEY", text)          # single-key path coexists
        self.assertIn("exit 2", text)                # both-set failure documented
        self.assertIn("/invocations", text)          # verified unguarded on 0.1.1

    def test_envs_doc_documents_contract(self):
        text = ENVS_DOC.read_text(encoding="utf-8")
        self.assertIn("`DSPARK_API_KEYS`", text)
        self.assertIn("`VLLM_API_KEY`", text)

    def test_no_key_material_committed(self):
        for path in (COMPOSE, ENV_EXAMPLE, *PROBES):
            text = path.read_text(encoding="utf-8")
            for match in re.findall(r"sk-[A-Za-z0-9_-]{6,}", text):
                self.assertIn(
                    match, ("sk-dspark-alice", "sk-dspark-bob", "sk-single-key"),
                    f"possible real key in {path.name}: {match}",
                )


if __name__ == "__main__":
    unittest.main()
