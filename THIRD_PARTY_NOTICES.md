# Third-party notices

This port translates code from, and its weights derive from, the projects below. Their licences travel with it.

## Dia2 — nari-labs/dia2 (code) and nari-labs/Dia2-2B (weights, tokenizer files)

Apache License 2.0 — Copyright Nari Labs. The full licence text is `LICENSES/Apache-2.0-nari-labs-dia2.txt`;
upstream ships no NOTICE file.

Translated files: `Dia2Core/{Config,Layers,Transformer,StateMachine,Sampling,Generator,Model}.swift`. They are
Swift / MLX translations of `dia2/config.py`, `dia2/core/{layers,cache,transformer,depformer,model}.py`,
`dia2/runtime/{state_machine,script_parser,guidance,sampler,generator,voice_clone,context}.py`,
`dia2/audio/grid.py`, `dia2/generation.py` and `dia2/engine.py` at `8687268`.

Changes from upstream:
- the language and framework (Python / PyTorch → Swift / MLX);
- RoPE angles computed per call instead of a fixed-length cache;
- the decoder cache viewed at its written length instead of masked;
- sampling through MLXRandom;
- prefix word timings supplied by the caller or estimated (`Dia2WordTiming`, new) instead of transcribed with
  whisper_timestamped;
- no CUDA-graph / torch.compile paths.

The weights are `nari-labs/Dia2-2B`, converted by `Tools/oracle-capture/convert.py`: keys unchanged, cast to the
tier dtype.

## Mimi — kyutai/mimi (weights)

CC-BY-4.0 — Mimi by Kyutai. https://huggingface.co/kyutai/mimi · https://creativecommons.org/licenses/by/4.0/

Changes: the transformers checkpoint re-keyed onto the moshi-swift module layout; attention q/k un-permuted from
transformers' half-split order and re-fused into `in_proj`; conv weights transposed to MLX layout; stored float32.

## moshi-swift — kyutai-labs/moshi-swift (code)

MIT License — Copyright (c) Kyutai. `Sources/Dia2Mimi/*.swift` are lifted from `MoshiLib` at `df64ffd`, with every
change marked `dia2:`:
- exact GELU in the transformer MLP;
- the current `Module.update` signature;
- whole-sequence `encodeFull` / `decodeFull` with plain causal attention.

The licence text is `Sources/Dia2Mimi/LICENSE-moshi-swift.txt`.

## Tokenizer

`tokenizer.json`, `vocab.json` and `merges.txt` ship in `nari-labs/Dia2-2B` (Apache-2.0) and are read by
[swift-transformers](https://github.com/huggingface/swift-transformers) (Apache-2.0).
