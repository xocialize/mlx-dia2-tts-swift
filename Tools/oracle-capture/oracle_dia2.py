"""Dia2 reference: the upstream nari-labs/dia2 runtime (ref/dia2, SHA in ref/shas.txt) on local weights.

Voice prefixes need word timestamps of the prefix audio; upstream gets them from `whisper_timestamped` (AGPL, lazily
imported). Here `transcribe_words` is replaced by a lookup into cues/refs.json — the same words, timed by
Whisper-large-v3-turbo (MLX) — so the AGPL package is never installed.

    from oracle_dia2 import load, generate
    dia = load("2B", device="mps")
    res = generate(dia, "[S1] Hello there.", seed=0, prefix_s1="cues/ref_s1.wav")
"""
import json, os, pathlib, random
import numpy as np, torch

from dia2 import Dia2, GenerationConfig, SamplingConfig
from dia2.generation import PrefixConfig
import dia2.runtime.voice_clone as VC

ROOT = pathlib.Path(os.environ.get("DIA_EVAL", "/Volumes/Satechi/Development/mlxengine-audio/WIP/dia-eval"))   # data stays in WIP (AB-D-0070)
MODELS = pathlib.Path("/Volumes/DEV_ARCHIVE/models")
REFS = json.load(open(ROOT / "cues/refs.json"))


def _transcribe_from_refs(audio_path, device, language=None):
    name = pathlib.Path(audio_path).stem
    if name not in REFS:
        raise KeyError(f"no precomputed words for {audio_path}")
    return [VC.WhisperWord(text=w["text"], start=w["start"], end=w["end"]) for w in REFS[name]["words"]]


VC.transcribe_words = _transcribe_from_refs
if hasattr(VC, "transcribe"):
    VC.transcribe = _transcribe_from_refs


def load(size="2B", device="mps", dtype="float32"):
    d = MODELS / f"nari-labs/Dia2-{size}"
    return Dia2.from_local(d / "config.json", d / "model.safetensors", device=device, dtype=dtype,
                           tokenizer_id=str(d), mimi_id=str(MODELS / "kyutai/mimi"))


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def generate(dia, script, seed=0, prefix_s1=None, prefix_s2=None, cfg_scale=2.0, temperature=0.8, top_k=50):
    """Upstream defaults (README / generation.py): cfg 2.0, audio temp 0.8 top-k 50, text temp 0.6 top-k 50."""
    seed_all(seed)
    prefix = PrefixConfig(speaker_1=prefix_s1, speaker_2=prefix_s2) if (prefix_s1 or prefix_s2) else None
    config = GenerationConfig(cfg_scale=cfg_scale, audio=SamplingConfig(temperature=temperature, top_k=top_k), prefix=prefix)
    return dia.generate(script, config=config)
