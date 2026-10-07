# mlx-dia2-tts-swift — measurements

M5 Max / 137 GB, macOS 27, mlx-swift 0.32.3. The parity gates are in `PORTING-SPEC.md`. This file records what the
shipping tier does: quality against upstream on E23's own cues, then speed and memory.

## Tier quality — the port vs upstream on E23's sets (2026-10-06)

E23's harness (`WIP/dia-eval/harness`) scored both systems with the same scorer:
- `run_dia2_v0.py`'s job list: speech × 3 seeds, clone2 × 2, dialogue × 2 × {plain, prefixed};
- rendered through the port by `dia2-gates --render-jobs` (bf16, GPU);
- scored by `score_dia.py`: Whisper-large-v3-turbo error, DNSMOS, CAM++, turn segmentation.

Upstream is E23's fp32 / MPS run of `nari-labs/dia2` @ `8687268`. Prefix word timings come from `cues/refs.json` in
both. The two systems draw from different random streams, so this compares two samples of the same distribution,
not the same takes.

| set | metric | upstream fp32 | **port bf16** | Δ (± SE) |
|---|---|---|---|---|
| single lines (30) | word err · fail · OVRL · trimmed s | 0.015 · 0 % · 3.18 · 4.72 | **0.008 · 0 % · 3.24 · 4.80** | |
| clone2, both prefixed (20) | cos to reference (min) · err | 0.798 (0.762) · 0.010 | **0.782 (0.731) · 0.000** | −0.016 ± 0.007 |
| scenes plain (20) | scene err · same / cross-spk · gap mean | 0.010 · 0.698 / 0.442 · 0.30 s | **0.016 · 0.695 / 0.446 · 0.28 s** | |
| scenes prefixed (20) | scene err · same / cross · S1 / S2 vs ref · gap | 0.007 · 0.713 / 0.401 · 0.689 / 0.732 · 0.32 s | **0.010 · 0.715 / 0.387 · 0.677 / 0.743 · 0.31 s** | S1 −0.012 ± 0.012 (paired by scene) · S2 +0.011 |

All 20 + 20 scenes had their four turns placed by the aligner. Upstream's own seed-to-seed spread on S1 is
sd 0.063 per scene.

**Reading.**
- Lines, plain scenes and prefixed scenes are indistinguishable from upstream within sampling noise.
- Against E23's gate C, the port's S1 (0.677) sits 0.060 below VoxCPM2's per-turn 0.737, where upstream sat
  0.048. That is a 0.012 move inside one standard error, and S2 moved +0.011 the other way.
- clone2 dropped by ≈ 2 SE; the fp32 run below separates bf16 from chance.

### bf16 vs fp32, and the transcript-estimated prefix timings

| system (bf16 unless noted) | clone2 cos (min) | scenes prefixed: err · same / cross · S1 / S2 vs ref · gap |
|---|---|---|
| port, refs.json word timings | 0.782 ± 0.006 (0.731) | 0.010 · 0.715 / 0.387 · 0.677 / 0.743 · 0.31 s |
| **port fp32**, refs.json timings | **0.788 ± 0.006 (0.725)** | — |
| **port, ESTIMATED timings** (`Dia2WordTiming`, transcript only) | **0.787 ± 0.005 (0.750)** | **0.013 · 0.697 / 0.382 · 0.676 / 0.724 · 0.30 s** |

- **bf16 costs nothing measurable.** fp32 vs bf16 clone2 differ by 0.006, inside one SE. The remaining gap to
  upstream (0.798) is sampling chance: the parity gates show every sampled distribution equals upstream's.
- **The estimate is a usable fallback** (paired by take, estimated − Whisper timings): clone2 +0.005 ± 0.007,
  S1 −0.001 ± 0.014, S2 −0.019 ± 0.013. The S2 trend is not significant, but timings from an aligner stay the
  preferred input (`metaData.referenceWords` / `speaker2Words`).

## Speed and memory — the package (`dia2-gates --validate`, 2026-10-06)

Quiet box (Edge closed), GPU, MLX pool at the engine's shipping cap (2 GiB; AB-L-0030 / AB-L-0155). Runs: a line; a
4-turn scene; the same scene with both speakers prefixed (estimated word timings, then again on the cached plan); a
12-turn scene; a 40-turn scene with both prefixes.

| tier | load | resident (MLX active) | RTF (3 s → 101 s takes) | peak activation | phys max | after unload |
|---|---|---|---|---|---|---|
| **bf16** | 0.24 s | **4 033 MB** | **0.53–0.60** (≈ 1.8× realtime; 0.59 at 101 s) | **≤ 1 214 MB** (947 MB at 3 s, 1 214 MB at 101 s) | 6 681 MB | MLX 0 |
| fp32 | 0.32 s | 7 694 MB | 0.97–1.72 | ≤ 1 711 MB | 10 456 MB | MLX 0 |

- **Activation is flat in take length** because Mimi's SEANet decoder streams (`Mimi.decodeChunked`, 50 steps per
  chunk). The first validate pass decoded whole takes and grew ~0.24 GB per second of audio: 1.28 / 3.51 / 6.53 GB
  at 3 / 11 / 23 s, i.e. ~30 GB at the 120 s cap. Bark found the same with EnCodec (E22).
- Prefixes cost their Mimi encode once; the plan is cached per (clips, transcripts, timings). The batched prefill
  replaces upstream's frame-by-frame warmup and is gated (g5).
- **Declared (`Dia2TTSPackage`, v0.1.1): bf16 4.50 GB resident + 3.00 GB activation, phys and pool-inclusive** —
  the fleet's convention (AB-L-0113 / AB-L-0155):
  - phys was 4 208 MB (4.41 GB) after load and 6 681 MB (7.01 GB) at the highest reading, so activation is 2.60 GB;
  - Dia2's own MLX working set is ≤ 1.21 GB of that, and the rest is the engine's 2 GiB recycling pool;
  - v0.1.0 had declared MLX-active numbers (4.30 + 1.50 GB), below the phys floor.

  bf16 is the one published tier (`mlx-community/Dia2-2B-bf16`). fp32 is the unpublished parity tier (a local
  `convert.py --dtype float32`): phys 7 870 → 10 456 MB, i.e. 8.25 + 2.71 GB, measured but not declared.
- Render-lane timings from the E23 job runs (RTF 0.9–1.5) are NOT speed numbers: those ran beside a browser at
  100 % CPU and a compile.

Cancel (`--cancel`, bf16): `CancellationError` arrives unwrapped **19 / 4 / 19 ms** after cancels at 0.3 s (prefix
encode / prefill), 2.5 s and 6.0 s. The same seeded request renders sample-identical before and after the cancels.

Output level varies with the seeded voice (peak −1 to −17 dBFS on the validate lines; all transcribe correctly), so
a consumer should level-normalise.
