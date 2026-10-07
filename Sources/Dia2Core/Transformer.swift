// Transformer.swift — dia2/core/transformer.py (TransformerDecoder, DecoderLayer) and dia2/core/depformer.py
// (Depformer, DepformerLayer, ScheduleAttention), ported 1:1.
//
// Decoder step: the step's 2 text streams go through MultiStreamEmbedding, each of the 32 audio channels through its
// own embedding, summed in channel order; pre-norm layers; final norm; action head (new-word / pad) and cb0 head.
// Depformer stage k (k = 0…30, codebook k+1): embed the previous codebook's token, add depformer_in[schedule[k]] of the
// decoder hidden, run the layers with ScheduleAttention (weights chosen by schedule[k], RoPE at position k, a cache over
// the frame's stages), norm, logits[k] cut to the audio vocabulary limit (min(pad, bos) = 2048).

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Decoder

public final class Dia2DecoderLayer: Module {
    @ModuleInfo(key: "pre_norm") var preNorm: Dia2RMSNorm
    @ModuleInfo(key: "attn") var attn: Dia2Attention
    @ModuleInfo(key: "post_norm") var postNorm: Dia2RMSNorm
    @ModuleInfo(key: "mlp") var mlp: Dia2Mlp

    init(_ cfg: Dia2Config) {
        let s = cfg.decoder
        _preNorm.wrappedValue = Dia2RMSNorm(s.nEmbd, eps: cfg.normEps)
        _attn.wrappedValue = Dia2Attention(cfg, dim: s.nEmbd)
        _postNorm.wrappedValue = Dia2RMSNorm(s.nEmbd, eps: cfg.normEps)
        _mlp.wrappedValue = Dia2Mlp(dim: s.nEmbd, hidden: s.nHidden, activations: cfg.linearActivations)
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray, cache: Dia2CacheSlot) -> MLXArray {
        let h = x + attn(preNorm(x), positions: positions, cache: cache)
        return h + mlp(postNorm(h))
    }
}

public final class Dia2TransformerDecoder: Module {
    @ModuleInfo(key: "audio_embeds") var audioEmbeds: [Embedding]
    @ModuleInfo(key: "text_embed") var textEmbed: Dia2MultiStreamEmbedding
    @ModuleInfo(key: "layers") var layers: [Dia2DecoderLayer]
    @ModuleInfo(key: "norm") var norm: Dia2RMSNorm
    @ModuleInfo(key: "action_head") var actionHead: Linear
    @ModuleInfo(key: "cb0_head") var cb0Head: Linear
    let cfg: Dia2Config

    init(_ cfg: Dia2Config) {
        self.cfg = cfg
        let s = cfg.decoder, d = cfg.data
        _audioEmbeds.wrappedValue = (0 ..< cfg.audioChannels).map { _ in Embedding(embeddingCount: d.audioVocabSize, dimensions: s.nEmbd) }
        _textEmbed.wrappedValue = Dia2MultiStreamEmbedding(vocab: d.textVocabSize, dim: s.nEmbd, padId: d.textPadTokenId, lowRankDim: s.lowRankDim)
        _layers.wrappedValue = (0 ..< s.nLayer).map { _ in Dia2DecoderLayer(cfg) }
        _norm.wrappedValue = Dia2RMSNorm(s.nEmbd, eps: cfg.normEps)
        _actionHead.wrappedValue = Linear(s.nEmbd, d.actionVocabSize, bias: false)
        _cb0Head.wrappedValue = Linear(s.nEmbd, d.audioVocabSize, bias: false)
    }

    public func makeCache(batch: Int, maxSteps: Int, dtype: DType) -> Dia2KVCache {
        Dia2KVCache(layers: layers.count, batch: batch, heads: cfg.decoder.kvHeads, maxSteps: maxSteps, headDim: cfg.decoder.headDim, dtype: dtype)
    }

    /// `forward_step`: tokens [B, C] (one step), position (same for every branch) →
    /// (normed hidden [B, 1, D], action logits [B, 1, 2] float32, cb0 logits [B, 1, V] float32).
    public func step(_ tokens: MLXArray, position: Int, cache: Dia2KVCache, computeDType: DType) -> (MLXArray, MLXArray, MLXArray) {
        let h = forward(tokens.expandedDimensions(axis: 1), startPosition: position, cache: cache, computeDType: computeDType)
        let hf = h.asType(actionHead.weight.dtype)
        return (h, actionHead(hf).asType(.float32), cb0Head(hf).asType(.float32))
    }

    /// T consecutive steps at positions start ..< start + T in one causal pass: tokens [B, T, C] → normed hidden
    /// [B, T, D]. Upstream's prefix warmup runs these one step at a time and discards the outputs; its inputs never
    /// depend on them (the action stream is forced), so one pass fills the cache with the same keys and values.
    public func forward(_ tokens: MLXArray, startPosition: Int, cache: Dia2KVCache, computeDType: DType) -> MLXArray {
        let (b, t) = (tokens.dim(0), tokens.dim(1))
        var hidden = textEmbed(tokens[0..., 0..., 0], tokens[0..., 0..., 1])
        for (idx, emb) in audioEmbeds.enumerated() {
            hidden = hidden + emb(tokens[0..., 0..., idx + 2]).asType(hidden.dtype)
        }
        var x = hidden.asType(computeDType)
        let positions = broadcast(MLXArray(Int32(startPosition) ..< Int32(startPosition + t)).reshaped([1, t]), to: [b, t])
        for (i, layer) in layers.enumerated() {
            x = layer(x, positions: positions, cache: cache.slots[i])
        }
        return norm(x)
    }
}

// MARK: - Depformer

public final class Dia2ScheduleAttention: Module {
    @ModuleInfo(key: "in_proj") var inProj: [Linear]
    @ModuleInfo(key: "out_proj") var outProj: [Linear]
    @ModuleInfo(key: "q_norm") var qNorm: Dia2RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Dia2RMSNorm
    let schedule: [Int]
    let heads: Int, kvHeads: Int, headDim: Int
    let rotary: Dia2Rotary?

    init(_ cfg: Dia2Config) {
        let s = cfg.depformer
        schedule = cfg.weightsSchedule
        heads = s.queryHeads; kvHeads = s.kvHeads; headDim = s.headDim
        let used = Set(schedule).sorted()
        precondition(used == Array(0 ..< used.count), "weights_schedule ids must be 0…n-1 (ModuleDict keys become array indices)")
        _inProj.wrappedValue = used.map { _ in Linear(s.nEmbd, 3 * s.queryHeads * s.headDim, bias: false) }
        _outProj.wrappedValue = used.map { _ in Linear(s.queryHeads * s.headDim, s.nEmbd, bias: false) }
        _qNorm.wrappedValue = Dia2RMSNorm(s.headDim, eps: cfg.normEps)
        _kNorm.wrappedValue = Dia2RMSNorm(s.headDim, eps: cfg.normEps)
        rotary = s.applyRope ? Dia2Rotary(headDim: s.headDim, minTimescale: cfg.ropeMinTimescale, maxTimescale: cfg.ropeMaxTimescale) : nil
    }

    func callAsFunction(_ x: MLXArray, stage: Int, cache: Dia2CacheSlot) -> MLXArray {
        let (b, t) = (x.dim(0), x.dim(1))
        let w = schedule[stage]
        let proj = inProj[w](x.asType(inProj[w].weight.dtype)).reshaped([b, t, 3, heads, headDim]).asType(x.dtype)
        var q = qNorm(proj[0..., 0..., 0], outDType: x.dtype)
        var k = kNorm(proj[0..., 0..., 1], outDType: x.dtype)
        let v = proj[0..., 0..., 2]
        if let rotary {
            let pos = MLXArray.full([b, t], values: MLXArray(Int32(stage)))
            q = rotary(q, positions: pos)
            k = rotary(k, positions: pos)
        }
        let (kc, vc) = cache.writeAndView(k.transposed(0, 2, 1, 3), v.transposed(0, 2, 1, 3))
        let y = MLXFast.scaledDotProductAttention(queries: q.transposed(0, 2, 1, 3), keys: kc, values: vc, scale: 1.0, mask: .none)
        let flat = y.transposed(0, 2, 1, 3).reshaped([b, t, heads * headDim])
        return outProj[w](flat.asType(outProj[w].weight.dtype)).asType(x.dtype)
    }
}

public final class Dia2DepformerLayer: Module {
    @ModuleInfo(key: "pre_norm") var preNorm: Dia2RMSNorm
    @ModuleInfo(key: "post_norm") var postNorm: Dia2RMSNorm
    @ModuleInfo(key: "self_attention") var selfAttention: Dia2ScheduleAttention
    @ModuleInfo(key: "mlp") var mlp: Dia2Mlp

    init(_ cfg: Dia2Config) {
        let s = cfg.depformer
        _preNorm.wrappedValue = Dia2RMSNorm(s.nEmbd, eps: cfg.normEps)
        _postNorm.wrappedValue = Dia2RMSNorm(s.nEmbd, eps: cfg.normEps)
        _selfAttention.wrappedValue = Dia2ScheduleAttention(cfg)
        _mlp.wrappedValue = Dia2Mlp(dim: s.nEmbd, hidden: s.nHidden, activations: s.mlpActivations)
    }

    func callAsFunction(_ x: MLXArray, stage: Int, cache: Dia2CacheSlot) -> MLXArray {
        let h = x + selfAttention(preNorm(x), stage: stage, cache: cache)
        return h + mlp(postNorm(h))
    }
}

public final class Dia2Depformer: Module {
    @ModuleInfo(key: "audio_embeds") var audioEmbeds: [Embedding]
    @ModuleInfo(key: "depformer_in") var depformerIn: [Linear]
    @ModuleInfo(key: "layers") var layers: [Dia2DepformerLayer]
    @ModuleInfo(key: "norm") var norm: Dia2RMSNorm
    @ModuleInfo(key: "logits") var logits: [Linear]
    let cfg: Dia2Config
    public let depth: Int
    public let vocabLimit: Int

    init(_ cfg: Dia2Config) {
        self.cfg = cfg
        let s = cfg.depformer, d = cfg.data
        precondition(!s.textEmbedding, "depformer text_embedding is not ported (both published checkpoints set it false)")
        depth = cfg.depth
        vocabLimit = min(d.audioPadTokenId, d.audioBosTokenId)
        _audioEmbeds.wrappedValue = (0 ..< depth).map { _ in Embedding(embeddingCount: d.audioVocabSize, dimensions: s.nEmbd) }
        _depformerIn.wrappedValue = Set(cfg.weightsSchedule).sorted().map { _ in Linear(cfg.decoder.nEmbd, s.nEmbd, bias: false) }
        _layers.wrappedValue = (0 ..< s.nLayer).map { _ in Dia2DepformerLayer(cfg) }
        _norm.wrappedValue = Dia2RMSNorm(s.nEmbd, eps: cfg.normEps)
        _logits.wrappedValue = (0 ..< depth).map { _ in Linear(s.nEmbd, d.audioVocabSize, bias: false) }
    }

    public func makeCache(batch: Int, dtype: DType) -> Dia2KVCache {
        Dia2KVCache(layers: layers.count, batch: batch, heads: cfg.depformer.kvHeads, maxSteps: depth, headDim: cfg.depformer.headDim, dtype: dtype)
    }

    /// `_forward_stage`: prevAudio [B] int, decoder hidden [B, 1, Dd] → logits [B, vocabLimit] float32.
    public func stage(_ k: Int, prevAudio: MLXArray, hidden: MLXArray, cache: Dia2KVCache, computeDType: DType) -> MLXArray {
        let w = cfg.weightsSchedule[k]
        let tokenEmb = audioEmbeds[k](prevAudio.expandedDimensions(axis: 1)).asType(computeDType)       // [B, 1, D]
        var x = depformerIn[w](hidden.asType(depformerIn[w].weight.dtype)).asType(computeDType) + tokenEmb
        for (i, layer) in layers.enumerated() {
            x = layer(x, stage: k, cache: cache.slots[i])
        }
        let h = norm(x)
        let out = logits[k](h.asType(logits[k].weight.dtype)).asType(.float32)                          // [B, 1, V]
        return out[0..., 0, ..<vocabLimit]
    }
}
