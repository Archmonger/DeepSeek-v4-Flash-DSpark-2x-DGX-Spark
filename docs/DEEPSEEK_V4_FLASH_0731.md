# DeepSeek V4 Flash 0731

`deepseek-ai/DeepSeek-V4-Flash-0731` supersedes the preview checkpoint while retaining the same `DeepseekV4ForCausalLM` and DSpark speculative-decoding structure.

## Checkpoint

- Repository: `deepseek-ai/DeepSeek-V4-Flash-0731`
- Tested revision: `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- Context: `1048576`
- DSpark block size: `5`
- Quantization metadata: FP8 weights
- Architecture: text-only causal language model

The published checkpoint has no vision processor, projector, or vision tower. Pair it with a separate multimodal sidecar when image input is required.

## Serving Profile

The default two-Spark profile uses MTP-5 probabilistic speculation, NVFP4 MLA KV cache, prefix caching, chunked prefill, asynchronous scheduling, CUDA graphs, and the `deepseek_v4` tokenizer, reasoning parser, and tool-call parser.

The model card does not ship a Jinja chat template. It includes an `encoding` package that defines message encoding and output parsing, including `low`, `high`, and `max` reasoning effort. Validate multi-turn role boundaries, reasoning separation, and tool calls after runtime upgrades because successful weight loading alone does not prove encoding compatibility.

Set `DSPARK_ENCODING_FILE` to the checkpoint's `encoding/encoding_dsv4.py` path inside the container when the runtime image predates the checkpoint. The launcher installs that encoder into vLLM before import, on both ranks. It also corrects pre-0731 tokenizer wrappers that mapped `low` reasoning effort to `high`. These changes are required for the 0731 `reasoning_content`, reasoning-effort, and tool-argument semantics.

## Benchmark Method

Run `scripts/benchmark-0731.py` against a warmed endpoint. The default sweep covers 256, 2K, 8K, 32K, and 128K prompt tokens at concurrency 1, 2, 4, and 6. It streams each response, records time to first token, and uses the API-reported token counts from naturally completed responses. It does not impose a server-side output limit.

```bash
python3 scripts/benchmark-0731.py \
  --base-url http://127.0.0.1:8888/v1 \
  --model deepseek-v4-flash-0731 \
  --output results/deepseek-v4-flash-0731.json
```

## Two-Spark Results

Results will be recorded from the pinned revision on two DGX Sparks after warmup.
