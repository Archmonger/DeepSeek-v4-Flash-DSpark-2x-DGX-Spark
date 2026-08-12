#!/usr/bin/env python3
"""compare-bench.py — Compare benchmark results (up to 3 scenarios)."""

import json, sys, os

def load(path):
    with open(path) as f:
        return json.load(f)

def main():
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")

    # Find available bench files
    no_patches = os.path.join(results_dir, "bench-baseline-no-patches.json")
    issue22_only = os.path.join(results_dir, "bench-baseline-issue22-only.json")
    all_patches = None
    for f in sorted(os.listdir(results_dir)):
        if f.startswith("bench-") and f.endswith(".json") and "baseline" not in f:
            all_patches = os.path.join(results_dir, f)

    available = []
    labels = []
    if os.path.exists(no_patches):
        available.append(load(no_patches))
        labels.append("No patches")
    if os.path.exists(issue22_only):
        available.append(load(issue22_only))
        labels.append("#22 only")
    if all_patches and os.path.exists(all_patches):
        available.append(load(all_patches))
        labels.append("All patches")

    if len(available) < 2:
        print("ERROR: Need at least 2 benchmark files in results/")
        print("Run:")
        print("  bash scripts/bench-baseline-no-patches.sh      # no patches")
        print("  bash scripts/bench-baseline-issue22-only.sh   # issue #22 only")
        print("  python3 scripts/bench-ttft.py                 # all patches")
        sys.exit(1)

    print(f"Comparing {len(available)} scenarios: {' vs '.join(labels)}")
    print()

    # Index by prompt_len
    by_label = []
    for data in available:
        by_label.append({r["prompt_len"]: r for r in data["results"] if r.get("num_ok", 0) > 0})

    # Find common lengths
    common = sorted(set.intersection(*[set(d) for d in by_label]))
    if not common:
        print("ERROR: No matching prompt lengths")
        sys.exit(1)

    # Header
    hdr = f"{'prompt':>8}"
    for label in labels:
        hdr += f" │ {label:>14}"
    print(hdr)
    sub = f"{'':>8}"
    for _ in labels:
        sub += f" │ {'p50':>7} {'p99':>9}"
    print(sub)
    print("─" * len(hdr))

    for pl in common:
        row = f"{pl:>8}"
        vals = []
        for d in by_label:
            r = d[pl]
            vals.append((r["latency_p50_ms"], r["latency_p99_ms"]))
            row += f" │ {r['latency_p50_ms']:>6.0f}ms {r['latency_p99_ms']:>8.0f}ms"
        print(row)

    # Delta table (first vs last)
    if len(available) >= 2:
        print()
        print(f"Delta: {labels[-1]} − {labels[0]}")
        print(f"{'prompt':>8} │ {'Δ P50':>10} │ {'Δ P99':>12} │ {'Verdict'}")
        print("─" * 55)
        for pl in common:
            b = by_label[0][pl]
            p = by_label[-1][pl]
            d50 = p["latency_p50_ms"] - b["latency_p50_ms"]
            d99 = p["latency_p99_ms"] - b["latency_p99_ms"]
            flag = "✅" if d99 < -500 else ("⚠️" if d99 > 500 else "≈")
            print(f"{pl:>8} │ {d50:>+9.0f}ms │ {d99:>+11.0f}ms │ {flag}")

if __name__ == "__main__":
    main()
