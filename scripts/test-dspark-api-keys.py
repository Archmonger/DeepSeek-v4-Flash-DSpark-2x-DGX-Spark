#!/usr/bin/env python3
"""CPU gates for DSPARK_API_KEYS multi-key auth (behavioral, stdlib only).

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
- a newline, carriage return, or dash placed after a newline in the value is
  rejected with exit 2 and the single-line message, in all four contexts (the
  single-line guard runs before tokenization, so a `\n-bad` value reports the
  single-line failure, not the dash-token failure);
- VLLM_API_KEY and DSPARK_API_KEYS both meaningful => exit 2 naming BOTH
  variables, in the compose entrypoint and in all three probe scripts. The
  server must never silently choose one (vLLM's CLI would win) and the probes
  must never guess which variable the server honoured;
- VLLM_API_KEY alone keeps the legacy single-key path unchanged;
- the three probe blocks are byte-identical (consistent parsing);
- Compose interpolates the key variables container-side (`$$`), so key text is
  never host-interpolated into shell source by Compose.

A ComposedHandoff layer additionally drives the REAL `docker compose
--env-file <stub> config --format json` render of docker-compose.dspark.yml
when the docker CLI is available and can render on the host, extracts the auth
block and exec tail from the rendered `command[2]`, runs them under the
rendered `environment`, and requires exactly one `--api-key` flag. It includes
two negative controls — the compose DSPARK_API_KEYS passthrough line removed,
and `$$` regressed to `$` in the entrypoint block — that must FAIL the chain.
Hosts without a working docker CLI skip the whole layer (never red).

run_bash is hermetic: the base env excludes VLLM_API_KEY and DSPARK_API_KEYS
(and any variable whose name contains "API_KEY"), so a hostile parent carrying
either/both variables cannot change any outcome; a case opts a variable in
first. The docker invocation is scrubbed the same way, and the handoff env is
the rendered compose `environment` alone.

The committed-key scan covers the compose file, .env.dspark.example, the three
probes, docs/ENVS.md, CHANGELOG.md, and this test file itself.

No GPU, no serve, stdlib only.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "docker-compose.dspark.yml"
ENV_EXAMPLE = ROOT / ".env.dspark.example"
ENVS_DOC = ROOT / "docs" / "ENVS.md"
CHANGELOG = ROOT / "CHANGELOG.md"
SELF = Path(__file__)
PROBES = (
    ROOT / "start-deepseek-v4-flash-dspark.sh",
    ROOT / "smoke-deepseek-v4-flash-dspark.sh",
    ROOT / "status-deepseek-v4-flash-dspark.sh",
)

PROBE_BEGIN = "# DSPARK_API_KEYS auth (begin)"
PROBE_END = "# DSPARK_API_KEYS auth (end)"

# Exact contract messages (identical in all four contexts, exit code 2).
BOTH_SET_MSG = (
    "error: VLLM_API_KEY and DSPARK_API_KEYS are both set; "
    "set exactly one of them"
)
SINGLE_LINE_MSG = "error: DSPARK_API_KEYS must be a single-line space-separated list"
DASH_REJECT_PREFIX = "error: DSPARK_API_KEYS token starts with '-'"

# Compose `$$` escapes stay `$$` in the rendered `command[2]` text; the regexes
# below tolerate both `$` and `$$` so one extractor serves source and render.
_AUTH_START = re.compile(r'_dspark_keys_set=0;[ \t]*case "\$+\{DSPARK_API_KEYS:-\}" in')
_AUTH_END = re.compile(r'API_KEY_ARGS=\(--api-key "\$+\{_dspark_keys\[@\]\}"\); fi;')

DOCKER = shutil.which("docker")


# --------------------------------------------------------------------------
# Extraction: get the real code out of the real files.
# --------------------------------------------------------------------------


def find_auth_block(text: str) -> str:
    """The entrypoint auth block inside `text` (compose source or rendered).

    Requires the block to be ONE physical line (ask #3: a folded-scalar-safe
    single compose line), and returns it verbatim (`$$` form in the source,
    `$$` form in the docker render — both decode to container-side `$`).
    """
    start = _AUTH_START.search(text)
    if start is None:
        raise AssertionError("entrypoint auth block not found")
    end = _AUTH_END.search(text, start.start())
    if end is None:
        raise AssertionError("entrypoint auth block not terminated by its ARGS line")
    block = text[start.start():end.end()]
    if "\n" in block:
        raise AssertionError(
            "entrypoint auth block must remain ONE physical compose line"
        )
    return block


def entrypoint_auth_block() -> str:
    """The compose entrypoint's auth line, still carrying Compose `$$` escapes."""
    return find_auth_block(COMPOSE.read_text(encoding="utf-8"))


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


def tail_from_text(text: str) -> str:
    """The `exec vllm serve ...` tail from a one-line rendered `command[2]`."""
    start = text.index("exec /usr/local/bin/vllm serve")
    return text[start:]


def probe_auth_block(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    begin = text.index(PROBE_BEGIN) + len(PROBE_BEGIN)
    end = text.index(PROBE_END, begin)
    return text[begin:end]


# --------------------------------------------------------------------------
# Shell helpers.
# --------------------------------------------------------------------------


def base_env(ambient=None):
    """Hermetic bash env: parent env minus ALL key material.

    VLLM_API_KEY and DSPARK_API_KEYS never reach a case through the parent
    (or through an injected hostile ambient), so a case must opt them in
    explicitly via env_kwargs. Anything whose name contains "API_KEY" is
    scrubbed (covers both variables plus any future variant).
    """
    env = dict(os.environ)
    if ambient:
        env.update(ambient)
    for name in [k for k in env if "API_KEY" in k]:
        del env[name]
    env.setdefault("PATH", "/usr/bin:/bin")
    env.setdefault("HOME", "/")
    return env


def run_bash(body: str, env_kwargs, cwd=None, ambient=None):
    """Run `body` in bash under a hermetic env.

    env_kwargs is the ONLY way key variables enter: a value sets it, None
    removes it (explicit absence even under a hostile parent).
    """
    env = base_env(ambient)
    for key, value in env_kwargs.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return subprocess.run(
        ["bash", "-c", body], capture_output=True, text=True, env=env, cwd=cwd
    )


def run_entrypoint(env_kwargs, cwd=None, ambient=None, block=None):
    """Run the real entrypoint auth block and print the resulting API_KEY_ARGS."""
    if block is None:
        block = entrypoint_auth_block()
    body = (
        block.replace("$$", "$") + "\n"
        + "printf 'COUNT=%s\\n' \"${#API_KEY_ARGS[@]}\";\n"
        + 'printf "<%s>\\n" "${API_KEY_ARGS[@]}"\n'
    )
    return run_bash(body, env_kwargs, cwd, ambient)


def run_probe(path: Path, env_kwargs, ambient=None):
    """Run a probe's real auth block and print the resulting AUTH_HEADER_ARGS."""
    body = (
        probe_auth_block(path) + "\n"
        + "printf 'COUNT=%s\\n' \"${#AUTH_HEADER_ARGS[@]}\";\n"
        + 'printf "<%s>\\n" "${AUTH_HEADER_ARGS[@]}"\n'
    )
    return run_bash(body, env_kwargs, ambient=ambient)


_VLLM_DUMPER = "python3 -c 'import sys; [print(\"A[\"+a+\"]\") for a in sys.argv[1:]]' "


def _decode_block_tail(block: str, tail: str):
    block = block.replace("$$", "$")
    tail = tail.replace("$$", "$").replace(
        "exec /usr/local/bin/vllm serve", _VLLM_DUMPER
    )
    return block, tail


def argv_of(env_kwargs, cwd=None, ambient=None, block=None, tail=None):
    """Run the real block + exec tail and return the argv handed to vllm serve."""
    if block is None:
        block = entrypoint_auth_block()
    if tail is None:
        tail = entrypoint_exec_tail()
    block, tail = _decode_block_tail(block, tail)
    r = run_bash(block + "\n" + tail, env_kwargs, cwd, ambient)
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
            self.assertIn(DASH_REJECT_PREFIX, r.stderr, value)
            self.assertIn("-bad", r.stderr, value)

    def test_newline_cr_dash_after_newline_exit2_single_line_msg(self):
        # ask #4: newline/CR anywhere in the value, and a dash placed after one,
        # must exit 2 with the single-line message (guard runs before read).
        for value in ("k1\nk2", "k1\rk2", "k1\n-bad", "k1\r-bad"):
            r = run_entrypoint({"DSPARK_API_KEYS": value})
            self.assertEqual(r.returncode, 2, (value, r.stderr))
            self.assertIn(SINGLE_LINE_MSG, r.stderr, (value, r.stderr))
            if value.endswith("-bad"):
                self.assertNotIn(DASH_REJECT_PREFIX, r.stderr, (value, r.stderr))

    def test_both_vars_named_and_exit_2(self):
        r = run_entrypoint({"VLLM_API_KEY": "vk", "DSPARK_API_KEYS": "k1 k2"})
        self.assertEqual(r.returncode, 2)
        self.assertIn(BOTH_SET_MSG, r.stderr)

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
            self.assertIn(DASH_REJECT_PREFIX, r.stderr, path.name)
            self.assertIn("-bad", r.stderr, path.name)

    def test_probes_newline_cr_dash_after_newline_exit2(self):
        # ask #4 in the probe contexts: exit 2 + single-line message, and the
        # dash-token message never fires for `\n-bad` (single-line guard wins).
        for path in PROBES:
            for value in ("k1\nk2", "k1\rk2", "k1\n-bad", "k1\r-bad"):
                r = run_probe(path, {"DSPARK_API_KEYS": value})
                self.assertEqual(r.returncode, 2, (path.name, value))
                self.assertIn(SINGLE_LINE_MSG, r.stderr, (path.name, value))
                if value.endswith("-bad"):
                    self.assertNotIn(DASH_REJECT_PREFIX, r.stderr,
                                     (path.name, value))

    def test_probes_exit_2_naming_both_vars(self):
        for path in PROBES:
            r = run_probe(path, {"VLLM_API_KEY": "vk", "DSPARK_API_KEYS": "k1"})
            self.assertEqual(r.returncode, 2, path.name)
            self.assertIn(BOTH_SET_MSG, r.stderr, path.name)

    def test_probes_vllm_only_unchanged(self):
        for path in PROBES:
            r = run_probe(path, {"VLLM_API_KEY": "vk"})
            self.assertEqual(r.returncode, 0, path.name)
            self.assertEqual(r.stdout,
                             "COUNT=2\n<-H>\n<Authorization: Bearer vk>\n",
                             path.name)


# --------------------------------------------------------------------------
# ComposedHandoff: real docker compose render + the real auth code.
# --------------------------------------------------------------------------


def render_compose(compose_text: str, env_text: str, ambient=None):
    """Render compose_text through the real docker CLI with env_text as --env-file.

    Runs `docker compose --env-file <stub.env> -f <temp compose> config
    --format json` in a temp dir with a key-scrubbed env (so a hostile parent
    can never leak keys into the render). Returns the parsed JSON object, or
    None when docker is unavailable / the render or JSON parse fails — the
    handoff layer then skips cleanly.
    """
    if not DOCKER:
        return None
    with tempfile.TemporaryDirectory() as td:
        compose_copy = Path(td, "docker-compose.dspark.yml")
        stub_env = Path(td, "stub.env")
        compose_copy.write_text(compose_text, encoding="utf-8")
        stub_env.write_text(env_text, encoding="utf-8")
        proc = subprocess.run(
            [DOCKER, "compose", "--env-file", str(stub_env), "-f",
             str(compose_copy), "config", "--format", "json"],
            capture_output=True, text=True, env=base_env(ambient), cwd=td,
        )
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout)
    except ValueError:
        return None


def rendered_env(render) -> dict:
    return render["services"]["vllm-dspark"]["environment"]


def rendered_command(render) -> list:
    return render["services"]["vllm-dspark"]["command"]


def env_from_render(render) -> dict:
    """The rendered compose `environment` as a runnable bash env.

    This is the ONLY carrier for handoff runs: exactly what the container
    would see, plus a usable PATH/HOME (the image defines PATH itself).
    """
    env = {}
    for key, value in rendered_env(render).items():
        if not isinstance(key, str):
            continue
        env[key] = "" if value is None else str(value)
    env.setdefault("PATH", os.environ.get("PATH", "/usr/bin:/bin"))
    env.setdefault("HOME", os.environ.get("HOME", "/"))
    return env


def handoff_entrypoint(block: str, env: dict):
    """Run the rendered container-side auth block under the rendered env."""
    body = (
        block.replace("$$", "$") + "\n"
        + "printf 'COUNT=%s\\n' \"${#API_KEY_ARGS[@]}\";\n"
        + 'printf "<%s>\\n" "${API_KEY_ARGS[@]}"\n'
    )
    return subprocess.run(["bash", "-c", body],
                          capture_output=True, text=True, env=env)


def handoff_argv(block: str, tail: str, env: dict):
    """Rendered block + exec tail under the rendered env → vllm argv."""
    block, tail = _decode_block_tail(block, tail)
    r = subprocess.run(["bash", "-c", block + "\n" + tail],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise AssertionError(f"handoff argv dump failed:\n{r.stderr}")
    return [ln[2:-1] for ln in r.stdout.splitlines()
            if ln.startswith("A[") and ln.endswith("]")]


class ComposedHandoff(unittest.TestCase):
    """Drive the REAL `docker compose config` render plus the REAL auth code.

    Skips cleanly when the docker CLI is missing or cannot render the compose
    file on the host — ci-validate.sh runners without docker must stay green.
    """

    HOSTILE = {"VLLM_API_KEY": "ambient-vk", "DSPARK_API_KEYS": "ambient-k1 ambient-k2"}
    STUB_ENV = (
        "DSPARK_API_KEYS=sk-alice sk-bob\n"
        "VLLM_API_KEY=\n"
        "DSPARK_MODEL=deepseek-ai/DeepSeek-V4-Flash-0731\n"
        "NCCL_SOCKET_IFNAME=eth0\n"
        "NCCL_IB_HCA=mlx5_0\n"
    )

    @classmethod
    def setUpClass(cls):
        cls.render = render_compose(COMPOSE.read_text(encoding="utf-8"),
                                    cls.STUB_ENV)
        if cls.render is not None:
            cls.command2 = rendered_command(cls.render)[2]
            cls.block = find_auth_block(cls.command2)
            cls.tail = tail_from_text(cls.command2)
            cls.env = env_from_render(cls.render)
        else:
            cls.command2 = cls.block = cls.tail = cls.env = None

    def setUp(self):
        if self.render is None:
            self.skipTest("docker compose cannot render on this host")

    def test_handoff_env_reaches_container(self):
        # The stub keys are what compose hands the container.
        self.assertEqual(self.env.get("VLLM_API_KEY"), "")
        self.assertEqual(self.env.get("DSPARK_API_KEYS"), "sk-alice sk-bob")

    def test_handoff_exactly_one_flag_carries_every_key(self):
        r = handoff_entrypoint(self.block, self.env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout, "COUNT=3\n<--api-key>\n<sk-alice>\n<sk-bob>\n",
                         r.stderr)
        argv = handoff_argv(self.block, self.tail, self.env)
        self.assertEqual(argv.count("--api-key"), 1, argv)
        i = argv.index("--api-key")
        self.assertEqual(argv[i + 1:i + 3], ["sk-alice", "sk-bob"])

    def test_handoff_passthrough_line_dropped_breaks_chain(self):
        # Negative control (a): remove the DSPARK_API_KEYS environment
        # passthrough line from the compose render; the YAML source's command
        # still contains the auth block, but the chain must detect NO flag.
        source = COMPOSE.read_text(encoding="utf-8")
        lines = [ln for ln in source.splitlines()
                 if not re.match(r'^\s*DSPARK_API_KEYS:\s*"\$\{DSPARK_API_KEYS:-\}"\s*$', ln)]
        variant = "\n".join(lines) + "\n"
        self.assertNotEqual(variant, source, "passthrough drop produced no edit")
        dropped = render_compose(variant, self.STUB_ENV)
        if dropped is None:
            self.skipTest("variant compose render failed on this host")
        dropped_env = env_from_render(dropped)
        # The stub still set the key; compose just never passed it through.
        self.assertNotIn("DSPARK_API_KEYS", dropped_env)
        argv = handoff_argv(self.block, self.tail, dropped_env)
        self.assertNotIn("--api-key", argv)

    def test_handoff_single_dollar_regression_caught(self):
        # Negative control (b): `$$` → `$` in the entrypoint block. Compose
        # then host-interpolates the stub keys straight into the shell source,
        # so the auth code no longer reads DSPARK_API_KEYS from the env —
        # the interpolation gate must detect the inlining.
        source = COMPOSE.read_text(encoding="utf-8")
        variant = source.replace("$${DSPARK_API_KEYS", "${DSPARK_API_KEYS")
        # Static first: the single-dollar refs are visible in the raw copy.
        variant_block = find_auth_block(variant)
        self.assertTrue(single_dollar_refs(variant_block),
                        "interpolation gate missed `$` in the compose block")
        # Rendered: compose inlines the stub keys; the shell `$` ref is gone.
        inline = render_compose(variant, self.STUB_ENV)
        if inline is None:
            self.skipTest("variant compose render failed on this host")
        seek = "${DSPARK_API_KEYS"
        self.assertIn(seek, self.command2,
                      "happy-path block lost its env ref (render changed?)")
        inline_command = rendered_command(inline)[2]
        self.assertIn("sk-alice", inline_command,
                      "variant render did not inline stub keys (gate vacuous)")
        self.assertNotIn(seek, inline_command,
                         "interpolation gate failed: keys inlined into shell source")

    def test_handoff_hermetic_under_hostile_parent(self):
        # Re-render with a hostile parent carrying BOTH key vars: the render
        # (and hence every outcome) must be byte-identical.
        hostile = render_compose(COMPOSE.read_text(encoding="utf-8"),
                                 self.STUB_ENV, ambient=self.HOSTILE)
        self.assertIsNotNone(hostile)
        self.assertEqual(rendered_env(hostile), rendered_env(self.render))
        self.assertEqual(rendered_command(hostile), rendered_command(self.render))
        self.assertEqual(env_from_render(hostile), self.env)


# --------------------------------------------------------------------------
# Ambient-parent env: hostile parent variants change nothing.
# --------------------------------------------------------------------------


class AmbientParent(unittest.TestCase):
    """A parent env carrying either/both key vars must change no outcome."""

    HOSTILE = ComposedHandoff.HOSTILE

    def test_entrypoint_outcomes_unchanged(self):
        cases = (
            {},
            {"DSPARK_API_KEYS": "k1"},
            {"DSPARK_API_KEYS": "k1 k2"},
            {"DSPARK_API_KEYS": "-bad"},
            {"DSPARK_API_KEYS": "k1\nk2"},
            {"VLLM_API_KEY": "vk"},
            {"VLLM_API_KEY": "vk", "DSPARK_API_KEYS": "k1"},
        )
        for env in cases:
            clean = run_entrypoint(env)
            hostile = run_entrypoint(env, ambient=self.HOSTILE)
            self.assertEqual(
                (hostile.returncode, hostile.stdout, hostile.stderr),
                (clean.returncode, clean.stdout, clean.stderr),
                env,
            )

    def test_probe_outcomes_unchanged(self):
        cases = (
            {},
            {"DSPARK_API_KEYS": "k1 k2"},
            {"DSPARK_API_KEYS": "-bad"},
            {"DSPARK_API_KEYS": "k1\nk2"},
            {"VLLM_API_KEY": "vk"},
            {"VLLM_API_KEY": "vk", "DSPARK_API_KEYS": "k1"},
        )
        for path in PROBES:
            for env in cases:
                clean = run_probe(path, env)
                hostile = run_probe(path, env, ambient=self.HOSTILE)
                self.assertEqual(
                    (hostile.returncode, hostile.stdout, hostile.stderr),
                    (clean.returncode, clean.stdout, clean.stderr),
                    (path.name, env),
                )

    def test_argv_outcomes_unchanged(self):
        for env in ({}, {"DSPARK_API_KEYS": "k1 k2"}, {"VLLM_API_KEY": "vk"}):
            self.assertEqual(argv_of(env, ambient=self.HOSTILE), argv_of(env), env)


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
        self.assertIn("hotfix-vllm-redact-api-key-log.sh", text)  # ask #2 docs

    def test_envs_doc_documents_contract(self):
        text = ENVS_DOC.read_text(encoding="utf-8")
        self.assertIn("`DSPARK_API_KEYS`", text)
        self.assertIn("`VLLM_API_KEY`", text)
        self.assertIn("hotfix-vllm-redact-api-key-log.sh", text)  # ask #2 row

    def test_no_key_material_committed(self):
        allow = ("sk-dspark-alice", "sk-dspark-bob", "sk-single-key")
        for path in (COMPOSE, ENV_EXAMPLE, *PROBES, ENVS_DOC, CHANGELOG, SELF):
            text = path.read_text(encoding="utf-8")
            for match in re.findall(r"sk-[A-Za-z0-9_-]{6,}", text):
                self.assertIn(match, allow,
                              f"possible real key in {path.name}: {match}")


if __name__ == "__main__":
    unittest.main()
