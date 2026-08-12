#!/usr/bin/env python3
"""Issue #24 repro — json_schema + thinking, run N times."""
import json, subprocess, sys, urllib.request

API = "http://127.0.0.1:8888/v1/chat/completions"

def make_req(schema, prompt, max_tokens=1500):
    return {
        "model": "deepseek-v4-flash-0731",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"thinking": True},
        "response_format": {"type": "json_schema", "json_schema": {"name": "r", "schema": schema}},
    }

def call(payload):
    req = urllib.request.Request(API, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())

schemas = [
    ("basic", {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
     "Return a JSON object with key city set to Paris."),
    ("person", {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "city": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}}, "required": ["name", "age", "city", "tags"]},
     "Required keys: name, age, city, tags. name: Alice, age 30, city Paris, tags [x, y]."),
]

fails = 0
for name, schema, prompt in schemas:
    for i in range(5):
        try:
            d = call(make_req(schema, prompt))
            c = d["choices"][0]["message"].get("content")
            try:
                json.loads(c); tag = "OK"
            except Exception as e:
                tag = f"INVALID({str(e)[:50]})"
            print(f"[{name}] run {i}: content={c!r} -> {tag}")
            if tag != "OK":
                fails += 1
        except Exception as e:
            print(f"[{name}] run {i}: transport error: {e}")
            fails += 1

print("\nTOTAL FAILS:", fails)
sys.exit(1 if fails else 0)
