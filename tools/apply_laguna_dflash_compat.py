#!/usr/bin/env python3
"""Port Poolside's Laguna DFlash decoder contract onto the buun tree.

The official Laguna-S DFlash GGUF contains seven Laguna-specific tensors beyond
plain DFlash: one stacked auxiliary-feature RMS norm and one attention output
gate for each of the six draft layers.  The generic loader ignores those
weights and then fails the tensor-count check (76 in the file versus 69 loaded).

This script is intentionally idempotent.  It is used by the temporary branch
workflow and is also useful when rebasing the port onto a newer buun commit.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count == 0:
        if new in text:
            return text
        raise RuntimeError(f"{label}: expected source fragment was not found")
    if count != 1:
        raise RuntimeError(f"{label}: source fragment occurred {count} times")
    return text.replace(old, new, 1)


def insert_after_once(text: str, marker: str, addition: str, label: str) -> str:
    if addition.strip() in text:
        return text
    return replace_once(text, marker, marker + addition, label)


def patch_arch_h(path: Path) -> None:
    text = path.read_text()
    text = insert_after_once(
        text,
        "    LLM_KV_NORM_BEFORE_RESIDUAL,\n",
        "    LLM_KV_DECODER_ARCH,\n",
        "llama-arch.h decoder metadata enum",
    )
    text = insert_after_once(
        text,
        "    LLM_TENSOR_ENC_OUTPUT_NORM,\n",
        "    LLM_TENSOR_ENC_AUX_NORM,\n",
        "llama-arch.h auxiliary norm tensor enum",
    )
    path.write_text(text)


def patch_arch_cpp(path: Path) -> None:
    text = path.read_text()
    text = insert_after_once(
        text,
        '    { LLM_KV_NORM_BEFORE_RESIDUAL,  "%s.norm_before_residual" },\n',
        '    { LLM_KV_DECODER_ARCH,          "%s.decoder_arch"         },\n',
        "llama-arch.cpp decoder metadata mapping",
    )
    text = insert_after_once(
        text,
        '    { LLM_TENSOR_ENC_OUTPUT_NORM,                        "enc.output_norm" },\n',
        '    { LLM_TENSOR_ENC_AUX_NORM,                           "enc.aux_norm" },\n',
        "llama-arch.cpp auxiliary norm name mapping",
    )
    text = insert_after_once(
        text,
        "    {LLM_TENSOR_ENC_OUTPUT_NORM,            {LLM_TENSOR_LAYER_OUTPUT,    GGML_OP_MUL}},\n",
        "    {LLM_TENSOR_ENC_AUX_NORM,               {LLM_TENSOR_LAYER_OUTPUT,    GGML_OP_MUL}},\n",
        "llama-arch.cpp auxiliary norm operator mapping",
    )
    path.write_text(text)


def patch_models_h(path: Path) -> None:
    text = path.read_text()
    if "bool decoder_laguna = false;" in text and "ggml_tensor * aux_norm = nullptr;" in text:
        return

    pattern = re.compile(r"(struct llama_model_dflash\s*:\s*public llama_model_base\s*\{\n)")
    match = pattern.search(text)
    if not match:
        raise RuntimeError("models.h: could not locate llama_model_dflash")

    fields = (
        "    // Laguna drafters have a target-specific decoder contract.\n"
        "    bool decoder_laguna = false;\n"
        "    ggml_tensor * aux_norm = nullptr;\n\n"
    )
    text = text[: match.end()] + fields + text[match.end() :]
    path.write_text(text)


def patch_dflash(poolside_path: Path, destination: Path) -> None:
    text = poolside_path.read_text()

    # Keep buun's helper declaration path and quantized-KV cache rotation support.
    text = insert_after_once(
        text,
        '#include "models.h"\n',
        '\n#include "llama-impl.h"\n',
        "dflash.cpp llama-impl include",
    )

    text = insert_after_once(
        text,
        "    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);\n",
        (
            "\n"
            "    // DFlash block size: default 16, overridable via GGUF. Set it here\n"
            "    // so only actual DFlash drafters advertise a non-zero block size.\n"
            "    hparams.dflash_block_size = 16;\n"
            "    ml.get_key(LLM_KV_DFLASH_BLOCK_SIZE, hparams.dflash_block_size, false);\n"
        ),
        "dflash.cpp block-size metadata",
    )

    swa_old = """                const bool    is_swa = hparams.is_swa(il);\n                const auto  * kv     = is_swa ? inp_attn_iswa->mctx->get_swa() : inp_attn_iswa->mctx->get_base();\n                ggml_tensor * k_idxs = is_swa ? inp_attn_iswa->get_k_idxs_swa() : inp_attn_iswa->get_k_idxs();\n                ggml_tensor * v_idxs = is_swa ? inp_attn_iswa->get_v_idxs_swa() : inp_attn_iswa->get_v_idxs();\n                ggml_build_forward_expand(gf, kv->cpy_k(ctx0, Kcur, k_idxs, il));\n                ggml_build_forward_expand(gf, kv->cpy_v(ctx0, Vcur, v_idxs, il));\n"""
    swa_new = """                const bool    is_swa = hparams.is_swa(il);\n                const auto  * kv     = is_swa ? inp_attn_iswa->mctx->get_swa() : inp_attn_iswa->mctx->get_base();\n                ggml_tensor * k_idxs = is_swa ? inp_attn_iswa->get_k_idxs_swa() : inp_attn_iswa->get_k_idxs();\n                ggml_tensor * v_idxs = is_swa ? inp_attn_iswa->get_v_idxs_swa() : inp_attn_iswa->get_v_idxs();\n                // Rotate injected K/V into the cache's quantized/rotated space.\n                ggml_tensor * k_rot  = is_swa ? inp_attn_iswa->self_k_rot_swa : inp_attn_iswa->self_k_rot;\n                ggml_tensor * v_rot  = is_swa ? inp_attn_iswa->self_v_rot_swa : inp_attn_iswa->self_v_rot;\n                if (k_rot) {\n                    Kcur = llama_mul_mat_hadamard(ctx0, Kcur, k_rot);\n                }\n                if (v_rot) {\n                    Vcur = llama_mul_mat_hadamard(ctx0, Vcur, v_rot);\n                }\n                ggml_build_forward_expand(gf, kv->cpy_k(ctx0, Kcur, k_idxs, il));\n                ggml_build_forward_expand(gf, kv->cpy_v(ctx0, Vcur, v_idxs, il));\n"""
    text = replace_once(text, swa_old, swa_new, "dflash.cpp iSWA KV rotation")

    dense_old = """            } else {\n                ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il));\n                ggml_build_forward_expand(gf, inp_attn->mctx->cpy_v(ctx0, Vcur, inp_attn->get_v_idxs(), il));\n            }\n"""
    dense_new = """            } else {\n                // Rotate injected K/V into the cache's quantized/rotated space.\n                if (inp_attn->self_k_rot) {\n                    Kcur = llama_mul_mat_hadamard(ctx0, Kcur, inp_attn->self_k_rot);\n                }\n                if (inp_attn->self_v_rot) {\n                    Vcur = llama_mul_mat_hadamard(ctx0, Vcur, inp_attn->self_v_rot);\n                }\n                ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il));\n                ggml_build_forward_expand(gf, inp_attn->mctx->cpy_v(ctx0, Vcur, inp_attn->get_v_idxs(), il));\n            }\n"""
    text = replace_once(text, dense_old, dense_new, "dflash.cpp dense KV rotation")

    destination.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poolside-dflash", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()

    root = args.root.resolve()
    patch_arch_h(root / "src/llama-arch.h")
    patch_arch_cpp(root / "src/llama-arch.cpp")
    patch_models_h(root / "src/models/models.h")
    patch_dflash(args.poolside_dflash, root / "src/models/dflash.cpp")

    print("Laguna DFlash compatibility port applied")


if __name__ == "__main__":
    main()
