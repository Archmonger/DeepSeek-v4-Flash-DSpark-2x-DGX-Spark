import argparse
import asyncio
import json
import statistics
import time
import urllib.request
from pathlib import Path


def request_json(url, body):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=3600) as response:
        return json.load(response)


def tokenize_url(base_url):
    return base_url.removesuffix("/v1") + "/tokenize"


def build_prompt(base_url, model, target):
    unit = "benchmark context datum "
    text = unit * max(1, target // 3)
    while True:
        count = request_json(tokenize_url(base_url), {"model": model, "prompt": text})["count"]
        if count >= target:
            return text
        text += unit * max(1, (target - count) // 3)


def stream_one(base_url, model, prompt):
    instruction = "\nReturn exactly 128 numbered lowercase English words, then stop."
    body = {"model": model, "messages": [{"role": "user", "content": prompt + instruction}], "stream": True, "stream_options": {"include_usage": True}, "temperature": 0.6, "top_p": 0.95}
    request = urllib.request.Request(f"{base_url}/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    first = None
    usage = None
    output = []
    output_chars = 0
    cancelled = False
    with urllib.request.urlopen(request, timeout=3600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            choices = event.get("choices") or []
            delta = choices[0].get("delta", {}) if choices else {}
            if first is None and (delta.get("content") or delta.get("reasoning_content")):
                first = time.perf_counter()
            reasoning = delta.get("reasoning_content") or ""
            content = delta.get("content") or ""
            output.extend((reasoning, content))
            output_chars += len(reasoning) + len(content)
            if event.get("usage"):
                usage = event["usage"]
            if output_chars >= 4096 or time.perf_counter() - started >= 180:
                cancelled = True
                break
    finished = time.perf_counter()
    measured = None if usage else request_json(tokenize_url(base_url), {"model": model, "prompt": "".join(output)})["count"]
    output_tokens = (usage or {}).get("completion_tokens", measured or 0)
    return {"ttft_s": (first or finished) - started, "elapsed_s": finished - started, "prompt_tokens": (usage or {}).get("prompt_tokens", 0), "output_tokens": output_tokens, "output_tok_s": output_tokens / max(0.001, finished - (first or finished)), "cancelled": cancelled}


async def run_case(base_url, model, prompt, concurrency):
    started = time.perf_counter()
    results = await asyncio.gather(*[asyncio.to_thread(stream_one, base_url, model, prompt) for _ in range(concurrency)])
    elapsed = time.perf_counter() - started
    total = sum(item["output_tokens"] for item in results)
    return {"concurrency": concurrency, "elapsed_s": elapsed, "aggregate_tok_s": total / max(0.001, elapsed), "median_ttft_s": statistics.median(item["ttft_s"] for item in results), "median_output_tok_s": statistics.median(item["output_tok_s"] for item in results), "requests": results}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    parser.add_argument("--model", default="deepseek-v4-flash-0731")
    parser.add_argument("--prompt-lengths", default="256,2048,8192,32768")
    parser.add_argument("--concurrency", default="1,2,4,6")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = {"model": args.model, "base_url": args.base_url, "cases": []}
    for prompt_length in [int(value) for value in args.prompt_lengths.split(",")]:
        prompt = await asyncio.to_thread(build_prompt, args.base_url, args.model, prompt_length)
        for concurrency in [int(value) for value in args.concurrency.split(",")]:
            case = await run_case(args.base_url, args.model, prompt, concurrency)
            case["target_prompt_tokens"] = prompt_length
            report["cases"].append(case)
            print(json.dumps(case, sort_keys=True), flush=True)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


asyncio.run(main())
