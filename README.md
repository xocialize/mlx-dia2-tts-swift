# mlx-dia2-tts-swift

[Dia2](https://github.com/nari-labs/dia2) (Nari Labs, Apache-2.0) on Apple silicon with
[MLX-Swift](https://github.com/ml-explore/mlx-swift), as an [MLXEngine](https://github.com/xocialize/mlx-engine-swift)
`tts` package. Checkpoint: [`nari-labs/Dia2-2B`](https://huggingface.co/nari-labs/Dia2-2B), converted to
[`mlx-community/Dia2-2B-bf16`](https://huggingface.co/mlx-community/Dia2-2B-bf16) (the package's default; the
engine materializes it).

Dia2 is a dialogue TTS. A script with `[S1]` / `[S2]` turns renders as **one take** in which two speakers trade
lines with conversational turn gaps: background conversations under a dubbed scene, generated dialogue, or one side
of an exchange spoken against the other party's audio. Each speaker can be conditioned on a voice prefix (a clip
plus its transcript).

Under the hood it is a delayed-streams model. A 28-layer decoder reads the script word by word through an action
stream ("new word" / "pad", 80 ms frames). A 4-layer depformer emits Kyutai Mimi's 32 codebooks per 12.5 Hz frame,
and Mimi decodes to 24 kHz.

| product | what |
|---|---|
| `Dia2Mimi` | Kyutai's Mimi codec, lifted from [kyutai-labs/moshi-swift](https://github.com/kyutai-labs/moshi-swift) (MIT); changes marked `dia2:` |
| `Dia2Core` | the port: decoder, depformer, the word/action state machine, CFG filter + sampling, voice prefixes, the generation loop |
| `MLXDia2TTS` | `Dia2TTSPackage`: the engine `ModelPackage` (canonical `TTSRequest` → 24 kHz mono `.wav`) |
| `dia2-gates` | parity gates against the upstream PyTorch runtime, plus GPU render lanes |

```swift
import MLXServeCore
import MLXDia2TTS

let engine = MLXServeEngine()
try await engine.register(Dia2TTSPackage.registration, configuration: Dia2TTSConfiguration())
try await engine.prepare(.tts)
let response = try await engine.run(TTSRequest(
    text: "[S1] Did you hear that? [S2] Hear what? It's the wind. [S1] No, it was a voice.",
    metaData: ["seed": .int(7)]))
```

**Voices.**
- `.auto` gives a fresh voice pair, fixed by `metaData.seed`.
- `.referenceAudio` + `referenceTranscript` is speaker 1's prefix.
- Speaker 2's prefix rides interim `metaData` keys (`speaker2Audio` as base64 `.wav`, plus `speaker2Transcript`) until the engine contract carries a second voice.
- Prefix word timings come from `metaData.referenceWords` / `speaker2Words` when the caller has an aligner's output; otherwise they are estimated from the transcript.
- A single prefix conditions the voice only weakly. Prefix **both** speakers for a scene.
- No preset voices, no emotion or duration control. Nonverbal tags such as `(laughs)` are accepted but rarely performed.

**Limits.** English only. One take holds ≤ 1 500 frames (120 s); a longer script is refused rather than truncated.
Output level varies with the seeded voice, so level-normalise downstream.

**Footprint (measured, bf16).** 4.5 GB resident + 3.0 GB activation as declared (phys_footprint, including the
engine's 2 GiB pool). Dia2's own working set is ≤ 1.2 GB, flat in take length. It runs at ≈ 1.8× realtime on an
M5 Max. On E23's cues the bf16 tier reproduces upstream's scene, line and cloning numbers within sampling noise
(`MEASUREMENTS.md`).

**Parity.** Tokenizer id-exact. Decoder within 1.1e-5 of the PyTorch reference. Every one of 2 673 + 1 584 sampled
distributions matches upstream within TV 1.1e-5, except one documented near-tie. Codes are token-exact under replay,
waveform within 114–118 dB. See `PORTING-SPEC.md`.

## Licences

Port code: MIT. It translates `nari-labs/dia2` (Apache-2.0) and lifts `kyutai-labs/moshi-swift` (MIT). Weights:
`nari-labs/Dia2-2B` is Apache-2.0; the bundled Mimi codec weights (`kyutai/mimi`) are CC-BY-4.0. See
`THIRD_PARTY_NOTICES.md`.

## Building and the gates

```bash
swift build -c release
swift test
```

The parity gates (`dia2-gates --gm --g1 --g2 --g3 --g5`) need an fp32 conversion
(`Tools/oracle-capture/convert.py OUT --dtype float32`) and goldens from the upstream runtime
(`Tools/oracle-capture/capture_goldens_dia2.py`). Point them at both with `DIA_EVAL` / `--weights` / `--goldens`.
