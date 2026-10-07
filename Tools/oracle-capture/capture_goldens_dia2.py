"""Parity goldens for the Dia2 Swift port, from the upstream nari-labs/dia2 runtime (ref/dia2 @ 8687268), fp32, CPU.
Written to _goldens/<size>/<name>/ as .npy + meta.json.

  g1_tokenizer   ids of the slow GPT-2 BPE tokenizer (use_fast=False, as upstream loads it) for edge strings, and
                 parse_script's entries (tokens + padding) for the golden scripts
  g2_steps       a plain generation, every transformer step recorded: the 2-branch step tokens fed in, the position,
                 action + cb0 logits (both branches), and the normed hidden of the first steps
  g3_replay      the same generation's every sampling call: kind (text / cb0 / dep<k>), the logits handed to the
                 sampler AFTER guidance + masking (branch 0), and the pick — the replay golden (AB-L-0203: the Swift
                 replay gate compares these distributions, not just the forced tokens); plus the final audio grid,
                 the undelayed tokens, the waveform and the word timestamps
  g4_mimi        Mimi (transformers MimiModel, kyutai/mimi): decode of g3's tokens → waveform; encode of
                 cues/ref_s1.wav → 32-codebook codes
  g5_prefix      the two-speaker prefix plan (S1 = ref_s1, S2 = ref_s2, words from cues/refs.json): entries,
                 new-word steps, aligned tokens; then a prefixed generation's replay (as g3)
usage (.venv-dia): capture_goldens_dia2.py [--size 2B] [--only g1,g2,g3,g4,g5]
"""
import argparse, json, os, pathlib
import numpy as np, torch

import oracle_dia2 as O
import dia2.runtime.generator as GEN
import dia2.runtime.guidance as GUI
import dia2.runtime.sampler as SAM
from dia2.runtime.script_parser import parse_script
from dia2.runtime.voice_clone import build_prefix_plan
from dia2.generation import PrefixConfig, GenerationConfig, SamplingConfig

ROOT = pathlib.Path(os.environ.get("DIA_EVAL", "/Volumes/Satechi/Development/mlxengine-audio/WIP/dia-eval"))
SCRIPT = "[S1] We should have left an hour ago. [S2] And whose fault is that?"


def save(gold, name, meta=None, **arrays):
    d = gold / name; d.mkdir(parents=True, exist_ok=True)
    for k, v in arrays.items():
        np.save(d / f"{k}.npy", np.ascontiguousarray(v))
    json.dump(dict(meta or {}, files=sorted(arrays)), open(d / "meta.json", "w"), indent=1, ensure_ascii=False)
    print(name, {k: (tuple(np.shape(v)), str(np.asarray(v).dtype)) for k, v in arrays.items()})


class Recorder:
    """Wraps the runtime's transformer step and the two samplers; records in call order."""
    def __init__(self, runtime):
        self.rt = runtime; self.steps = []; self.draws = []
        self._tstep = runtime.transformer_step
        def tstep(tokens, positions, cache):
            h, a, c, cache = self._tstep(tokens, positions, cache)
            self.steps.append(dict(tokens=tokens.detach().cpu().numpy().copy(), pos=int(positions[0, 0]),
                                   action=a.detach().float().cpu().numpy().copy(), cb0=c.detach().float().cpu().numpy().copy(),
                                   hidden=h.detach().float().cpu().numpy().copy() if len(self.steps) < 4 else None))
            return h, a, c, cache
        runtime.transformer_step = tstep
        self._sample_token, self._sample_audio = GEN.sample_token, GEN.sample_audio_logits
        rec = self
        def sample_token(logits, temp, top_k=0):
            pick = rec._sample_token(logits, temp=temp, top_k=top_k)
            rec.draws.append(("text", logits.detach().float().cpu().numpy().reshape(-1).copy(), int(pick.reshape(-1)[0])))
            return pick
        def sample_audio(logits, temp, top_k):
            pick = rec._sample_audio(logits, temp, top_k)
            rec.draws.append(("audio", logits.detach().float().cpu().numpy().reshape(-1).copy(), int(pick.reshape(-1)[0])))
            return pick
        GEN.sample_token, GEN.sample_audio_logits = sample_token, sample_audio

    def restore(self):
        self.rt.transformer_step = self._tstep
        GEN.sample_token, GEN.sample_audio_logits = self._sample_token, self._sample_audio


def run_recorded(dia, script, seed, prefix=None):
    runtime = dia._ensure_runtime()
    rec = Recorder(runtime)
    try:
        res = O.generate(dia, script, seed=seed, **(prefix or {}))
    finally:
        rec.restore()
    return res, rec


def save_replay(gold, name, res, rec, meta):
    # kinds: 0 = text (action, 2-wide), 1 = cb0 (2050-wide, pad/bos masked), 2 = depformer stage (2048-wide)
    kind_of = lambda k, l: 0 if k == "text" else (1 if l.shape[0] > 2048 else 2)
    kinds = np.array([kind_of(k, l) for k, l, _ in rec.draws], dtype=np.int32)
    picks = np.array([p for _, _, p in rec.draws], dtype=np.int32)
    text_logits = np.stack([l for k, l, _ in rec.draws if kind_of(k, l) == 0]).astype(np.float32)
    cb0_logits = np.stack([l for k, l, _ in rec.draws if kind_of(k, l) == 1]).astype(np.float32)
    dep_logits = np.stack([l for k, l, _ in rec.draws if kind_of(k, l) == 2]).astype(np.float32)
    words = [w for w, _ in res.timestamps]; times = np.array([t for _, t in res.timestamps], dtype=np.float32)
    save(gold, name, dict(meta, n_steps=len(rec.steps), n_draws=len(picks), words=words),
         draw_kinds=kinds, draw_picks=picks, text_logits=text_logits, cb0_logits=cb0_logits, dep_logits=dep_logits,
         tokens=res.audio_tokens[0].detach().cpu().numpy().astype(np.int32),
         waveform=res.waveform.detach().float().cpu().numpy().reshape(-1), word_times=times)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--size", default="2B"); ap.add_argument("--only", default="g1,g2,g3,g4,g5")
    a = ap.parse_args(); todo = a.only.split(",")
    torch.set_num_threads(12)
    gold = ROOT / "_goldens" / a.size.lower(); gold.mkdir(parents=True, exist_ok=True)
    dia = O.load(a.size, device="cpu")
    runtime = dia._ensure_runtime()
    tok = runtime.tokenizer
    if "g1" in todo:
        strings = ["[S1] We should have left an hour ago.", "[S2] And whose fault is that? (laughs)", "(clears throat) Right.",
                   "Hello, world! It's 3:45 — isn't it?", "Café naïve résumé", "  spaces   and\ttabs ", "(sings) La la la",
                   "[S1] Hi. [S2] Hey. [S1] Bye.", "Don’t stop", 'Wait <break time="1.5s"/> now']
        ids = [tok.encode(s, add_special_tokens=False) for s in strings]
        scripts = [SCRIPT, "[S1] (laughs) Okay, okay, I deserved that one.", 'Wait <break time="1.5s"/> now [S2] fine']
        ents = [[dict(tokens=e.tokens, text=e.text, padding=e.padding) for e in parse_script([s], tok, runtime.constants, runtime.frame_rate)]
                for s in scripts]
        c = runtime.constants
        json.dump(dict(strings=strings, ids=ids, scripts=scripts, entries=ents,
                       constants=dict(card=c.card, new_word=c.new_word, pad=c.pad, bos=c.bos, zero=c.zero, spk1=c.spk1, spk2=c.spk2,
                                      audio_pad=c.audio_pad, audio_bos=c.audio_bos), frame_rate=runtime.frame_rate),
                  open(gold / "g1_tokenizer.json", "w"), indent=0, ensure_ascii=False)
        print("g1_tokenizer", len(strings), "strings,", len(scripts), "scripts")
    if "g2" in todo or "g3" in todo:
        res, rec = run_recorded(dia, SCRIPT, seed=0)
        steps = rec.steps
        save(gold, "g2_steps", dict(script=SCRIPT, seed=0),
             tokens=np.stack([s["tokens"][:, :, 0] for s in steps]).astype(np.int32),
             positions=np.array([s["pos"] for s in steps], dtype=np.int32),
             action=np.stack([s["action"].reshape(2, -1) for s in steps]), cb0=np.stack([s["cb0"].reshape(2, -1) for s in steps]),
             hidden=np.stack([s["hidden"].reshape(2, -1) for s in steps[:4]]))
        save_replay(gold, "g3_replay", res, rec, dict(script=SCRIPT, seed=0, cfg_scale=2.0, text_temp=0.6, audio_temp=0.8, top_k=50))
    if "g4" in todo:
        tokens = np.load(gold / "g3_replay/tokens.npy")
        mimi = runtime.mimi
        wav = mimi.decode(torch.from_numpy(tokens.astype(np.int64))[None]).float().cpu().numpy().reshape(-1)
        from dia2.runtime.audio_io import load_mono_audio, encode_audio_tokens
        ref = load_mono_audio(str(ROOT / "cues/ref_s1.wav"), mimi.sample_rate)
        codes = encode_audio_tokens(mimi, ref).cpu().numpy().astype(np.int32)
        save(gold, "g4_mimi", dict(sample_rate=mimi.sample_rate), tokens=tokens, waveform=wav, ref_audio=ref.astype(np.float32), ref_codes=codes)
    if "g5" in todo:
        pc = PrefixConfig(speaker_1=str(ROOT / "cues/ref_s1.wav"), speaker_2=str(ROOT / "cues/ref_s2.wav"))
        plan = build_prefix_plan(runtime, pc)
        ents = [dict(tokens=e.tokens, text=e.text, padding=e.padding) for e in plan.entries]
        script = "[S1] Fine. Let's just go."
        res, rec = run_recorded(dia, script, seed=1, prefix=dict(prefix_s1=pc.speaker_1, prefix_s2=pc.speaker_2))
        json.dump(dict(entries=ents, new_word_steps=plan.new_word_steps, aligned_frames=plan.aligned_frames, script=script),
                  open(gold / "g5_prefix_plan.json", "w"), indent=0, ensure_ascii=False)
        save_replay(gold, "g5_prefix", res, rec, dict(script=script, seed=1, prefix_steps=sum(1 for s in rec.steps if s["pos"] < plan.aligned_frames)))
        np.save(gold / "g5_prefix/aligned_tokens.npy", plan.aligned_tokens.cpu().numpy().astype(np.int32))
