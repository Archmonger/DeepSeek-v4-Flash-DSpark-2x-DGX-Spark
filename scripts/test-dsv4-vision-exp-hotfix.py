#!/usr/bin/env python3
"""CPU tests for Vision-Exp image layout + fail-closed hotfix text patches."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "patches"))

from vision_exp.image_processor import (  # noqa: E402
    IMAGE,
    IMAGE_END,
    IMAGE_START,
    as_pil,
    build_image_block,
    grid_tokens,
    is_unregistered_router_bias,
    is_vision_exp_weight_name,
    vision_args_from_config,
)

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

_spec = importlib.util.spec_from_file_location(
    "hotfix_dsv4_vision_exp",
    ROOT / "patches" / "hotfix-dsv4-vision-exp.py",
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
patch_encoding_text = _mod.patch_encoding_text
patch_model_text = _mod.patch_model_text
patch_dspark_text = _mod.patch_dspark_text
ENC_MARK = _mod.ENC_MARK
ENC_ROLE_MARK = _mod.ENC_ROLE_MARK
MODEL_MARK = _mod.MODEL_MARK
DSPARK_MARK = _mod.DSPARK_MARK


class VisionExpLayoutTest(unittest.TestCase):
    def test_grid_tokens_square_is_bounded(self):
        n_h, n_w, n_tok = grid_tokens(756, 756, 14, 3)
        self.assertEqual((n_h, n_w), (18, 18))
        self.assertLessEqual(n_tok, 384)

    @unittest.skipUnless(HAS_TORCH, "torch not installed on this host")
    def test_build_image_block_starts_and_ends(self):
        types, perm = build_image_block(4, 4, start_pos=3)
        self.assertEqual(int(types[0].item()), IMAGE_START)
        self.assertEqual(int(types[-1].item()), IMAGE_END)
        self.assertGreater(int((types == IMAGE).sum()), 0)
        self.assertEqual(int(perm.numel()), 16)

    @unittest.skipUnless(HAS_TORCH, "torch not installed on this host")
    def test_pil_to_patches_respects_max_tokens(self):
        from PIL import Image

        from vision_exp.image_processor import pil_to_patches

        args = vision_args_from_config(
            type("C", (), {"vision_max_n_token": 384, "hidden_size": 4096})()
        )
        image = Image.new("RGB", (2048, 2048), (20, 40, 80))
        patches, n_h, n_w, n_llm_h, n_llm_w = pil_to_patches(image, args)
        types, perm = build_image_block(n_llm_h, n_llm_w, start_pos=0)
        self.assertEqual(patches.shape[0], n_h * n_w)
        self.assertLessEqual(types.numel(), 384)
        self.assertEqual(int(perm.numel()), int((types == IMAGE).sum()))

    def test_as_pil_accepts_pil_hwc_chw_and_dict(self):
        from PIL import Image

        rgb = Image.new("RGB", (4, 6), (10, 20, 30))
        got = as_pil(rgb)
        self.assertEqual(got.size, (4, 6))
        self.assertEqual(got.getpixel((0, 0)), (10, 20, 30))
        wrapped = as_pil({"image": rgb})
        self.assertEqual(wrapped.size, (4, 6))
        try:
            import numpy as np
        except ImportError:
            return
        hwc = np.zeros((6, 4, 3), dtype="uint8")
        hwc[..., 0] = 11
        self.assertEqual(as_pil(hwc).getpixel((0, 0)), (11, 0, 0))
        chw = np.zeros((3, 6, 4), dtype="uint8")
        chw[1] = 22
        self.assertEqual(as_pil(chw).getpixel((0, 0)), (0, 22, 0))

    def test_vision_weight_names_bypass_stacked_w1(self):
        self.assertTrue(is_vision_exp_weight_name("aligner.w1.bias"))
        self.assertTrue(is_vision_exp_weight_name("vision.blocks.0.mlp.w1.weight"))
        self.assertTrue(is_vision_exp_weight_name("image_start"))
        self.assertFalse(is_vision_exp_weight_name("layers.0.ffn.w1.weight"))
        self.assertFalse(is_vision_exp_weight_name("model.layers.0.ffn.w1.weight"))

    def test_hash_layer_gate_bias_is_skipped(self):
        routed = {"layers.3.ffn.gate.e_score_correction_bias"}
        self.assertTrue(
            is_unregistered_router_bias(
                "layers.0.ffn.gate.e_score_correction_bias", routed
            )
        )
        self.assertFalse(
            is_unregistered_router_bias(
                "layers.3.ffn.gate.e_score_correction_bias", routed
            )
        )
        self.assertFalse(
            is_unregistered_router_bias(
                "layers.0.ffn.gate.e_score_correction_bias_vl",
                {"layers.0.ffn.gate.e_score_correction_bias_vl"},
            )
        )
        self.assertTrue(
            is_unregistered_router_bias("model.layers.0.ffn.gate.bias_vl", {})
        )
        self.assertFalse(
            is_unregistered_router_bias(
                "model.layers.0.ffn.gate.e_score_correction_bias_vl",
                {"model.layers.0.ffn.gate.e_score_correction_bias_vl"},
            )
        )

    def test_eagle3_aux_target_is_causal_lm_model(self):
        """DSpark/Eagle3 need get_language_model().model (the inner tower)."""

        class Inner:
            def __init__(self):
                self.aux = None
                self.layers = [None] * 43

            def _set_aux_hidden_state_layers(self, layers):
                self.aux = layers

            def embed_input_ids(self, x):
                return x

        class CausalLM:
            def __init__(self):
                self.model = Inner()

            def get_language_model(self):
                return self

        lm = CausalLM()
        parent_ref = lm.get_language_model()
        self.assertTrue(hasattr(parent_ref, "model"))
        self.assertIs(parent_ref.model, lm.model)
        parent_ref.model._set_aux_hidden_state_layers((41, 42, 43))
        self.assertEqual(lm.model.aux, (41, 42, 43))

    def test_hash_moe_keeps_raw_input_ids(self):
        text = (ROOT / "patches" / "vision_exp" / "apply.py").read_text()
        self.assertIn("requires_raw_input_tokens = True", text)
        self.assertIn("multimodal_embeddings", text)
        self.assertIn("_merge_multimodal_embeddings", text)


class VisionExpHotfixTextTest(unittest.TestCase):
    def test_model_inject_is_idempotent(self):
        src = (
            "class DeepseekV4MoE:\n    pass\n\n"
            "class DeepseekV4ForCausalLM:\n    pass\n"
        )
        first, st1 = patch_model_text(src)
        second, st2 = patch_model_text(first)
        self.assertEqual(st1, "applied")
        self.assertEqual(st2, "skipped")
        self.assertIn(MODEL_MARK, first)
        self.assertEqual(first, second)

    def test_model_missing_class_is_drift(self):
        _, status = patch_model_text("class Other:\n    pass\n")
        self.assertTrue(status.startswith("drift"))

    def test_encoding_relaxes_placeholder_checks(self):
        src = '''
IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"

def _validate_no_image_sp_tokens(msg):
    content = msg.get("content")
    if isinstance(content, str) and IMAGE_PLACEHOLDER in content:
        raise ValueError("bad")
    reasoning_content = msg.get("reasoning_content")
    if isinstance(reasoning_content, str) and IMAGE_PLACEHOLDER in reasoning_content:
        raise ValueError("bad-reason")

def _process_image_blocks(blocks):
    text = blocks[0].get("text") or ""
    if IMAGE_PLACEHOLDER in text:
        raise ValueError("bad-text")
'''
        updated, status = patch_encoding_text(src)
        self.assertEqual(status, "applied")
        self.assertIn(ENC_MARK, updated)
        self.assertIn(ENC_ROLE_MARK, updated)
        skipped, status2 = patch_encoding_text(updated)
        self.assertEqual(status2, "skipped")
        self.assertEqual(updated, skipped)
        ns: dict = {}
        exec(compile(updated, "encoding.py", "exec"), ns)
        placeholder = ns["IMAGE_PLACEHOLDER"]
        ns["_validate_no_image_sp_tokens"](
            {"role": "user", "content": placeholder}
        )
        ns["_validate_no_image_sp_tokens"](
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,xx"}},
                    {"type": "text", "text": "what is this?"},
                ],
            }
        )
        with self.assertRaises(ValueError) as system_err:
            ns["_validate_no_image_sp_tokens"](
                {
                    "role": "system",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "http://x/y.png"}}
                    ],
                }
            )
        self.assertIn("user messages only", str(system_err.exception))
        with self.assertRaises(ValueError) as assistant_err:
            ns["_validate_no_image_sp_tokens"](
                {"role": "assistant", "content": placeholder}
            )
        self.assertIn("assistant", str(assistant_err.exception))
        ns["_validate_no_image_sp_tokens"]({"role": "system", "content": "text only"})

    def test_dspark_remaps_bias_vl_before_lookup(self):
        src = '''
class DSparkDeepseekV4ForCausalLM:
    def load_weights(self, weights):
        params_dict = {}
        for name, loaded_weight in weights:
            if False:
                pass
            else:
                if name.endswith(".ffn.gate.bias"):
                    name = name.replace(
                        ".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"
                    )
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", None)
'''
        updated, status = patch_dspark_text(src)
        self.assertEqual(status, "applied")
        self.assertIn(DSPARK_MARK, updated)
        self.assertIn(".ffn.gate.bias_vl", updated)
        self.assertIn("if name not in params_dict:", updated)
        skipped, status2 = patch_dspark_text(updated)
        self.assertEqual(status2, "skipped")
        self.assertEqual(updated, skipped)
        _, drift = patch_dspark_text("def load_weights(self, weights): pass\n")
        self.assertTrue(drift.startswith("drift"))


class VisionExpComposeWiringTest(unittest.TestCase):
    def test_limit_mm_is_json_not_bare_image_eq(self):
        text = (ROOT / "docker-compose.dspark.yml").read_text()
        self.assertNotIn(
            "--limit-mm-per-prompt ${LIMIT_MM_PER_PROMPT:-image=8}",
            text,
        )
        self.assertIn('LIMIT_MM_ARGS=(--limit-mm-per-prompt "$${LIMIT_MM_JSON}")', text)
        self.assertIn('"$${LIMIT_MM_ARGS[@]}"', text)
        self.assertIn("${SERVED_MODEL_NAME:-deepseek-v4-flash-vision-exp}", text)

    def test_worker_vision_exp_sync_does_not_nest(self):
        text = (ROOT / "start-deepseek-v4-flash-dspark.sh").read_text()
        self.assertIn('scp -r "$SCRIPT_DIR/patches/vision_exp/."', text)
        self.assertIn("rm -rf '${REMOTE_WORKER_DIR}/patches/vision_exp'", text)


if __name__ == "__main__":
    unittest.main()
