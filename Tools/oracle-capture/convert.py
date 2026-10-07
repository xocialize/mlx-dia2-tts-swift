"""Convert nari-labs/Dia2-{2B,1B} + kyutai/mimi into the layout the Swift port loads.

  model.safetensors   the Dia2 transformer + depformer, keys exactly as upstream (`transformer.*`, `depformer.*`), cast to
                      --dtype (bfloat16 shipping / float32 parity)
  mimi.safetensors    kyutai/mimi (transformers MimiModel — the exact checkpoint Dia2 decodes with), re-keyed onto the
                      module paths of the lifted kyutai-labs/moshi-swift Mimi (Sources/Dia2Mimi), float32:
                        · HF splits Kyutai's fused attention in_proj into q/k/v and PERMUTES q and k from interleaved to
                          half-split rotary (transformers' convert_mimi_checkpoint_to_pytorch.py); moshi-swift's RoPE is
                          interleaved, so q and k are un-permuted and re-fused
                        · conv weights torch (O, I, K) → MLX (O, K, I); transposed convs torch (I, O, K) → (O, K, I)
                        · codebooks keep embed_sum / cluster_usage / initialized (the module derives the embedding)
  config.json         Dia2's config.json verbatim (+ "mimi": {"num_codebooks": 32}) · tokenizer files (vocab.json,
                      merges.txt, tokenizer.json, tokenizer_config.json, special_tokens_map.json, added_tokens.json)
usage: convert.py <out_dir> [--size 2B|1B] [--dtype bfloat16|float32]
"""
import argparse, json, pathlib, re, shutil
import torch
from safetensors.torch import load_file, save_file

MODELS = pathlib.Path("/Volumes/DEV_ARCHIVE/models")
TOKENIZER_FILES = ["vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"]


def unpermute(w, n_heads):
    """Inverse of transformers' permute(w, n) = w.view(n, d/n/2, 2, d2).transpose(1, 2).reshape(d, d2)."""
    d1, d2 = w.shape
    return w.view(n_heads, 2, d1 // n_heads // 2, d2).transpose(1, 2).reshape(d1, d2)


def mimi_key(k):
    """HF MimiModel key → moshi-swift module key (None = drop)."""
    m = re.match(r"(encoder|decoder)\.layers\.(\d+)\.(.*)$", k)
    if m:
        side, idx, rest = m.group(1), int(m.group(2)), m.group(3)
        convtr = side == "decoder" and idx in (2, 5, 8, 11)
        rest = rest.replace("conv.", "convtr.convtr." if convtr else "conv.conv.", 1)
        rest = rest.replace("block.1.", "block.0.").replace("block.3.", "block.1.")
        if idx == 0:
            return f"{side}.init_conv1d.{rest}"
        if idx == 14:
            return f"{side}.final_conv1d.{rest}"
        if side == "encoder":
            for j, e in enumerate((1, 4, 7, 10)):
                if idx == e: return f"encoder.layers.{j}.residuals.0.{rest}"
                if idx == e + 2: return f"encoder.layers.{j}.downsample.{rest}"
        else:
            for j, dd in enumerate((2, 5, 8, 11)):
                if idx == dd: return f"decoder.layers.{j}.upsample.{rest}"
                if idx == dd + 1: return f"decoder.layers.{j}.residuals.0.{rest}"
        raise KeyError(k)
    m = re.match(r"(encoder|decoder)_transformer\.layers\.(\d+)\.(.*)$", k)
    if m:
        side, n, rest = m.groups()
        rest = (rest.replace("mlp.fc1", "gating.linear1").replace("mlp.fc2", "gating.linear2")
                .replace("self_attn.o_proj", "self_attn.out_proj").replace("input_layernorm", "norm1")
                .replace("post_attention_layernorm", "norm2").replace("self_attn_layer_scale", "layer_scale_1")
                .replace("mlp_layer_scale", "layer_scale_2"))
        return f"{side}_transformer.transformer.layers.{n}.{rest}"
    if k == "downsample.conv.weight":
        return "downsample.conv.conv.conv.weight"
    if k == "upsample.conv.weight":
        return "upsample.convtr.convtr.convtr.weight"
    m = re.match(r"quantizer\.(semantic|acoustic)_residual_vector_quantizer\.(.*)$", k)
    if m:
        which = "rvq_first" if m.group(1) == "semantic" else "rvq_rest"
        rest = m.group(2)
        rest = re.sub(r"^layers\.(\d+)\.codebook\.", r"vq.layers.\1._codebook.", rest)
        rest = rest.replace("embed_sum", "embedding_sum").replace("._codebook.initialized", "._codebook._initialized")
        return f"quantizer.{which}.{rest}"
    raise KeyError(k)


def convert_mimi(hf):
    cfg = json.load(open(hf / "config.json"))
    heads = cfg["num_attention_heads"]
    sd = load_file(str(hf / "model.safetensors"))
    out = {}
    fused = {}
    for k, v in sd.items():
        v = v.float()
        m = re.match(r"((?:encoder|decoder)_transformer\.layers\.\d+)\.self_attn\.([qkv])_proj\.weight$", k)
        if m:
            fused.setdefault(m.group(1), {})[m.group(2)] = v
            continue
        nk = mimi_key(k)
        if nk.endswith(".convtr.weight"):
            v = v.permute(1, 2, 0)                  # (I, O, K) → (O, K, I)
        elif v.ndim == 3:
            v = v.permute(0, 2, 1)                  # (O, I, K) → (O, K, I)
        out[nk] = v.contiguous()
    for prefix, qkv in fused.items():
        side, n = re.match(r"(\w+)_transformer\.layers\.(\d+)", prefix).groups()
        w = torch.cat([unpermute(qkv["q"], heads), unpermute(qkv["k"], heads), qkv["v"]], dim=0)
        out[f"{side}_transformer.transformer.layers.{n}.self_attn.in_proj.weight"] = w.contiguous()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--size", default="2B")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"]); a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    src = MODELS / f"nari-labs/Dia2-{a.size}"
    dt = torch.bfloat16 if a.dtype == "bfloat16" else torch.float32
    sd = {k: v.to(dt).contiguous() for k, v in load_file(str(src / "model.safetensors")).items()}
    save_file(sd, str(out / "model.safetensors"), metadata={"format": "mlx", "source": f"nari-labs/Dia2-{a.size}"})
    print("model", len(sd), sum(v.numel() for v in sd.values()) / 1e6, "M params")
    mimi = convert_mimi(MODELS / "kyutai/mimi")
    save_file(mimi, str(out / "mimi.safetensors"), metadata={"format": "mlx", "source": "kyutai/mimi (moshi-swift layout)"})
    print("mimi", len(mimi), sum(v.numel() for v in mimi.values()) / 1e6, "M params")
    cfg = json.load(open(src / "config.json")); cfg["mimi"] = {"num_codebooks": 32, "source": "kyutai/mimi"}; cfg["dtype"] = a.dtype
    json.dump(cfg, open(out / "config.json", "w"), indent=1)
    for f in TOKENIZER_FILES:
        if (src / f).exists(): shutil.copy(src / f, out / f)
    print("wrote", out)
