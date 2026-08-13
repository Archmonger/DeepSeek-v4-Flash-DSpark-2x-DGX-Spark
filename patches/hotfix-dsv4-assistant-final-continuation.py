#!/usr/bin/env python3
"""Hotfix: emit a generation header when a request ends with an assistant turn.

Symptom
-------
An agent harness gets stuck emitting empty turns: 1-2 generated tokens with
`finish_reason: "stop"`, no tool call, and fragments of hallucinated markup
(`<result observation="no-op"></content>`, stray `</parameter>` / `</name>` /
`<value>`), or a sentence fragment that continues the previous turn instead of
answering. Server metrics during a live incident: 6 of 37 requests generated
<= 10 tokens, all `stop`, zero `length` — the model chose to stop rather than
being truncated.

Cause
-----
`render_message()` appends the generation header (`ASSISTANT_SP_TOKEN` plus the
thinking token) only when the trailing message is `user` or `developer`:

    elif messages[index].get("role") in ["user", "developer"]:

When the request's `messages` array ends with an **assistant** message, that
branch is skipped, the turn is closed with EOS, and the prompt ends on a bare
EOS with no header. The model is asked to generate from a closed, turnless
state. Measured on this recipe:

    assistant-final: '...to the reviewer for a fresh verdict:<|end of sentence|>'
    user-final:      '...verdict:<EOS><|User|>continue<|Assistant|><think>'

It is self-sustaining: the harness records the resulting empty turn, so the next
request also ends with an assistant message. One bad turn locks the loop;
inserting any user-role message breaks it instantly. Requests reach this shape
whenever a harness retries after a mid-stream error and re-sends the partial
assistant turn, which is why the same prompt works for a long time and then
does not.

Why a header and not `wo_eos`
-----------------------------
`encoding_dsv4.py` also has `assistant_msg_wo_eos_template`, and reopening the
trailing turn looks like a tempting one-line fix. It is wrong. Measured on one
prompt via /v1/completions, generated tokens:

    trailing turn   | stock (EOS) | wo_eos (reopen) | EOS + header
    partial         | 32          | 208             | 260
    complete        | 16 (fragment)| 1 (EMPTY, dead)| 140 (healthy)

Reopening a *complete* assistant turn just moves the dead state: the model has
nothing left to add and emits EOS immediately. Appending a fresh generation
header is correct for both shapes, and matches `add_generation_prompt=True`
semantics that OpenAI-compatible clients already assume.

Scope
-----
Only the final message of a request is affected, and only when it is an
assistant turn. Every other rendering path, including consecutive assistant
messages mid-transcript, is untouched.

Idempotent. Patches the encoding module the server actually loads, so it must
run AFTER the compose entrypoint copies `encoding_dsv4.py` into place.
"""
from pathlib import Path

TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/tokenizers/deepseek_v4_encoding.py"
)
MARK = "[assistant-final-hotfix]"

OLD = (
    "    elif messages[index].get(\"role\") in [\"user\", \"developer\"]:\n"
    "        # Normal generation: append Assistant + thinking token\n"
)
NEW = (
    "    elif messages[index].get(\"role\") in [\"user\", \"developer\"] or (\n"
    f"        # {MARK} A request may legitimately end with an assistant turn\n"
    "        # (harness retry, continuation). Without a generation header the\n"
    "        # prompt ends on a bare EOS and the model generates from a dead\n"
    "        # state: immediate EOS, or raw DSML markup emitted as text.\n"
    "        messages[index].get(\"role\") == \"assistant\"\n"
    "        and index == len(messages) - 1\n"
    "    ):\n"
    "        # Normal generation: append Assistant + thinking token\n"
)


def main() -> None:
    if not TARGET.exists():
        # The encoder is copied in from the checkpoint by the compose
        # entrypoint; if that did not happen there is nothing to patch.
        print(f"[assistant-final-hotfix] not found, skipping: {TARGET}")
        return

    src = TARGET.read_text()
    if MARK in src:
        print("[assistant-final-hotfix] already applied")
        return
    if OLD not in src:
        raise SystemExit(f"[assistant-final-hotfix] anchor not found in {TARGET}")

    TARGET.write_text(src.replace(OLD, NEW, 1))
    print(f"[assistant-final-hotfix] patched {TARGET}")

    import importlib.util

    spec = importlib.util.spec_from_file_location("enc_check", TARGET)
    enc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enc)
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "A finished answer."},
    ]
    tail = enc.encode_messages(msgs, "thinking", reasoning_effort="high")[-40:]
    if enc.thinking_start_token in tail:
        print("[assistant-final-hotfix] verified: generation header present")
    else:
        print(f"[assistant-final-hotfix] WARNING: no header in tail: {tail!r}")


if __name__ == "__main__":
    main()
