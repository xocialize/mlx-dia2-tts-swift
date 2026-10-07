# mlx-dia2-tts-swift — porting spec

Dia2 (Nari Labs, Apache-2.0) → Swift-MLX, checkpoint `nari-labs/Dia2-2B`. Evaluation, the niche question and the go
decision live in `mlxengine-audio/Docs/ENHANCEMENTS.md` **E23** (AB-D-0110); this file is the port's contract and its
gate record.

## Upstream

| piece | source | licence | pinned |
|---|---|---|---|
| algorithm (state machine, script parser, CFG filter, sampling, prefix plans, generation loop) | `nari-labs/dia2` `dia2/runtime/*.py`, `dia2/engine.py`, `dia2/generation.py`, `dia2/audio/grid.py` | Apache-2.0 | `8687268` (2025-11-29) |
| decoder / depformer | `dia2/core/{transformer,depformer,layers,cache}.py` | Apache-2.0 | same |
| weights | `nari-labs/Dia2-2B` `model.safetensors`, `config.json`, tokenizer files | Apache-2.0 | main |
| codec | Kyutai Mimi: weights `kyutai/mimi` (transformers `MimiModel`, the checkpoint upstream decodes with); Swift module code from `kyutai-labs/moshi-swift` `MoshiLib` | weights CC-BY-4.0, code MIT | moshi-swift `df64ffd` |

Not ported: `whisper_timestamped` (AGPL), which upstream lazily imports to time a prefix's words. Word timings
come from the caller (an aligner's output) or `Dia2WordTiming.estimate`. CUDA graphs / `torch.compile` are not
ported either; they are upstream performance paths with identical numerics.

## Architecture

| stage | module | shape (2B) |
|---|---|---|
| step input | `Dia2MultiStreamEmbedding` (2 text streams, shared table, two projections, second stream dropped where pad) + 32 audio embeddings, summed | text vocab 49 280, audio vocab 2 050 |
| decoder | 28 pre-norm layers: GQA attention (16 q / 8 kv heads × 128, per-head q/k RMSNorm, half-split RoPE, scale 1.0) + gated MLP (SiLU · linear, 6 144) | 2 048 wide |
| heads | action (new-word / pad, 2) and cb0 (2 050) from the final norm | |
| depformer | per frame, 31 stages (codebooks 1–31): stage k embeds codebook k−1's token, adds `depformer_in[schedule[k]]`(decoder hidden), 4 layers whose attention weights are chosen by `weights_schedule[k]` (5 sets), RoPE at position k, a per-frame cache; logits cut to 2 048 | 1 024 wide |
| codec | Mimi decoder (32 codebooks, 12.5 Hz → 24 kHz); the encoder for voice prefixes | |
| runtime | 2 CFG branches (branch 1 always reads text (zero, pad)); per frame: action draw → state machine → cb0 draw → 31 depformer draws; delay pattern 16 / 18 × 31; EOS at the script's last new-word + 24 frames | |

## Key contracts

- Module paths equal the checkpoint keys (`transformer.layers.N.attn.q_proj`, `depformer.layers.N.self_attention.in_proj.K` …).
  `WeightIO.apply` compares exact key sets, then calls `update(verify: [.noUnusedKeys, .shapeMismatch])`.
  Mimi's derived `expandedWeight` and its EMA codebook buffers are named exceptions.
- `Tools/oracle-capture/convert.py` re-keys `kyutai/mimi` onto moshi-swift paths. transformers permutes Mimi's
  q/k for half-split RoPE, so the converter un-permutes them and re-fuses `in_proj`. Conv weights become MLX
  `(O, K, I)`. Dia2's weights are cast to the tier dtype; norms stay float32 in every tier.
- **CFG is a filter.** `guided = lerp(uncond, cond, scale)` decides which entries survive (≥ its 50th-largest
  value). The CONDITIONAL logits of the survivors are what gets sampled. Upstream's `topk(sorted=False)[..., -1]`
  is the k-th largest on its CPU/CUDA kernels.
- Sampling is `softmax(l / T)` → top-k → renormalise → multinomial, which equals a categorical draw over `l / T`
  restricted to its top-k. Port draws use MLXRandom, so they reproduce the distribution, not torch's stream.
  Parity on sampled stages is by **replay** (below).
- Upstream quirks kept on purpose:
  - The loop restarts at `aligned_frames − 1` after a prefix warmup, so that position is decoded twice (the decoder cache appends).
  - `first_word_frame` keys on `main == new_word`, which the second-stream multiplexer never emits, so the crop is the start step.
  - Speaker 2's prefix new-word steps get speaker 1's frame count but not the `+3`.
  - The last ~18 frames of the prefix audio are never fed: generation overwrites them in the delayed grid.
- Python `round` is half-to-even: break padding and prefix frame rounding use `.toNearestOrEven`.
- RoPE angles are computed per call. Upstream caches 1 500 positions, which a prefixed take could exceed.
- **Prefix warmup as one batched prefill.** Upstream steps the decoder once per prefix frame and discards the
  outputs. Those inputs depend only on the prefix codes and the forced state machine, so the port builds all of
  them first and fills the cache in causal chunks of 256 frames. That is the same keys and values up to summation
  order, and g5 holds it to TV 6.2e-6.
- **Streamed codec decode.** Mimi's decoder transformer runs over the whole take (plain causal, as transformers'
  MimiModel), then the causal SEANet decoder streams in 50-step chunks through its conv state (`decodeChunked`).
  Whole-take decode is linear in memory (~0.24 GB per second of audio). The streamed one is flat and matches it to
  2.6e-6.

## Gates (`swift run -c release dia2-gates …`, run from the package dir)

Goldens: `Tools/oracle-capture/capture_goldens_dia2.py` over the upstream runtime (fp32, CPU, seed 0 / 1).
Prefix words come from `cues/refs.json` via the oracle's Whisper-turbo timings. Parity gates run on the CPU stream
in fp32 against the fp32 conversion.

| gate | what | result (2026-10-06) |
|---|---|---|
| gm Mimi | decode of g3's tokens (whole-take and streamed in chunks of 7 / 50 steps); encode of a 10.2 s clip | decode max \|Δ\| 6.3e-6, **SNR 114.5 dB**; chunked 114.7 dB at both chunk sizes (≤ 2.6e-6 from whole-take, same length); encode **100 %** of 32 codebooks |
| g1 tokenizer | 10 edge strings (cues, tags, digits, accents, curly quote, break tag); `parse_script` on 3 scripts; token ids | **10/10 id-exact**; **3/3 entries identical**; ids identical |
| g2 decoder | 81 recorded steps × 2 branches through one cache | hidden 4.3e-6; action 4.8e-6; cb0 **1.05e-5** |
| g3 replay | a 12-word two-speaker script: every one of 2 673 draws (81 text, 81 cb0, 2 511 depformer) compared before the forced pick, then grid, undelay, crop, Mimi, timestamps | logits ≤ 7.5e-5; sampling-distribution TV ≤ **1.1e-5**; **one near-tie** (below); tokens exact; waveform **114.5 dB**; timestamps exact |
| g5 prefixed replay | two-speaker prefix (10.2 s + 8.1 s, 53 words): plan entries + new-word steps; 230-frame warmup (batched prefill) + 48 frames, 1 584 draws | plan identical; TV ≤ **6.2e-6**, no ties; tokens exact; waveform **117.8 dB**; timestamps exact |

**The near-tie (g3 draw 1812, depformer stage 28).** Upstream's and the port's guided logits straddle the CFG
filter's 50th place, where the gap is **4.3e-6**, the order of the logits' own fp32 noise. One entry is in one
survivor set and not the other, which puts that draw's TV at 5.5e-3. Exactly-equal survivor sets are not
achievable at that gap. The gate reports such draws when the 50th-place gap is ≤ 1e-4 and holds every other draw
to TV < 1e-3.

Live lanes (GPU; results in `MEASUREMENTS.md`):
- `--validate [--quant bf16|fp32]`: the package through load → 6 runs (3 – 101 s, prefixed and not) → refusals →
  unload, with measured memory.
- `--cancel`: three mid-run cancels plus a sample-identical seeded re-run.
- `--render-jobs FILE [--system NAME] [--estimate-words]`: E23's job list into the layout `harness/score_dia.py`
  scores.

Offline: `swift test` runs 16 conformance and unit tests (manifest, MAT-1..5, CAN-1..3, request plane, state-machine
trace, word-timing estimate, CFG filter).

## Lessons this port paid for

1. **A forced-token replay gate must compare the distributions** (AB-L-0203, from Bark). Here every one of 4 257
   draws is compared before the forced pick.
2. **A boundary near-tie is not a port defect, and loosening the threshold is not the fix.** Measure the gap the
   filter cut at and gate on it. Raising the TV bound to 6e-3 would have hidden a real error of that size anywhere
   else.
3. **HF Mimi's attention is plain causal on the sdpa path**, not the 250-step sliding window its config names.
   Whole-sequence encode/decode (`encodeFull` / `decodeFull`) follows it.
4. **A whole-take codec decode makes activation linear in take length.** Here it was ~0.24 GB per second of audio,
   so ~30 GB at the 120 s cap. Bark found the same with EnCodec. Measure activation at two take lengths before
   declaring a footprint; a causal codec streams.
