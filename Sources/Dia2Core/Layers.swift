// Layers.swift — dia2/core/layers.py + dia2/core/cache.py, ported 1:1. Module paths match the checkpoint
// (`attn.q_proj`, `attn.q_norm`, `mlp.wi`, `text_embed.main_proj`, …).
//
// Numerics follow upstream: every RMSNorm runs in float32 (upstream builds them `dtype=torch.float32`); projections take
// their input in the weight dtype (upstream casts to float32 before each Linear — the fp32 tier is that exactly);
// attention uses scale 1.0 because Q and K are RMS-normalised per head first; RoPE is the half-split ("rotate_half")
// form over timescales min · (max / min)^(2i / d).

import Foundation
import MLX
import MLXFast
import MLXNN

/// `nn.RMSNorm(dim, eps, dtype=torch.float32)`; the result is cast back to `outDType` (the compute dtype).
public final class Dia2RMSNorm: Module {
    public let weight: MLXArray
    let eps: Float

    public init(_ dim: Int, eps: Float) {
        weight = MLXArray.ones([dim], dtype: .float32)
        self.eps = eps
    }

    public func callAsFunction(_ x: MLXArray, outDType: DType? = nil) -> MLXArray {
        let y = MLXFast.rmsNorm(x.asType(.float32), weight: weight.asType(.float32), eps: eps)
        return y.asType(outDType ?? x.dtype)
    }
}

/// `RotaryEmbedding` — half-split rotation, inverse frequencies 1 / (min · (max / min)^(2i / d)) computed in float32
/// with the same ops as upstream. Angles are computed per call (upstream's cos/sin cache stops at max_context_steps).
struct Dia2Rotary {
    let invFreq: MLXArray      // [d / 2] float32

    init(headDim: Int, minTimescale: Double, maxTimescale: Double) {
        let half = headDim / 2
        let fraction = (2.0 * MLXArray(0 ..< half).asType(.float32)) / Float(headDim)
        let ratio = Float(maxTimescale / minTimescale)
        let timescale = Float(minTimescale) * MLX.pow(MLXArray(ratio), fraction)
        invFreq = 1.0 / timescale
    }

    /// x [B, T, H, D], positions [B, T] (int) → rotated, in x's dtype.
    func callAsFunction(_ x: MLXArray, positions: MLXArray) -> MLXArray {
        let freqs = positions.asType(.float32).expandedDimensions(axis: -1) * invFreq     // [B, T, D/2]
        let emb = concatenated([freqs, freqs], axis: -1)                                   // [B, T, D]
        let cosT = cos(emb).expandedDimensions(axis: 2).asType(x.dtype)
        let sinT = sin(emb).expandedDimensions(axis: 2).asType(x.dtype)
        let halves = split(x, parts: 2, axis: -1)
        let rotated = concatenated([-halves[1], halves[0]], axis: -1)
        return (x * cosT) + (rotated * sinT)
    }
}

/// `CacheSlot` — preallocated keys / values [B, H, maxSteps, D] and a fill length. Upstream attends over all maxSteps
/// positions with an additive mask (finfo.min) on the unwritten tail; slicing to the written length is the same
/// computation (masked positions get exactly zero weight) without the wasted work.
public final class Dia2CacheSlot {
    var keys: MLXArray
    var values: MLXArray
    public private(set) var length = 0

    init(batch: Int, heads: Int, maxSteps: Int, headDim: Int, dtype: DType) {
        keys = MLXArray.zeros([batch, heads, maxSteps, headDim], dtype: dtype)
        values = MLXArray.zeros([batch, heads, maxSteps, headDim], dtype: dtype)
    }

    func reset() { length = 0 }

    /// k, v [B, H, step, D] → the written prefix of the cache, [B, H, length, D].
    func writeAndView(_ k: MLXArray, _ v: MLXArray) -> (MLXArray, MLXArray) {
        let step = k.dim(2)
        precondition(length + step <= keys.dim(2), "Dia2 KV cache overflow (\(length + step) > \(keys.dim(2)))")
        keys[0..., 0..., length ..< (length + step), 0...] = k.asType(keys.dtype)
        values[0..., 0..., length ..< (length + step), 0...] = v.asType(values.dtype)
        length += step
        return (keys[0..., 0..., ..<length, 0...], values[0..., 0..., ..<length, 0...])
    }
}

public final class Dia2KVCache {
    let slots: [Dia2CacheSlot]
    init(layers: Int, batch: Int, heads: Int, maxSteps: Int, headDim: Int, dtype: DType) {
        slots = (0 ..< layers).map { _ in Dia2CacheSlot(batch: batch, heads: heads, maxSteps: maxSteps, headDim: headDim, dtype: dtype) }
    }
    func reset() { slots.forEach { $0.reset() } }
}

/// Activation by upstream name (`_get_activation`).
func dia2Activation(_ name: String, _ x: MLXArray) -> MLXArray {
    switch name.lowercased() {
    case "silu", "swish", "swiglu": return silu(x)
    case "gelu", "geglu": return gelu(x)
    case "relu": return relu(x)
    case "linear": return x
    default: preconditionFailure("unsupported activation \(name)")
    }
}

/// `Mlp` — wi projects to [2, hidden] (gate, up), act0(gate) · act1(up), wo back.
public final class Dia2Mlp: Module {
    @ModuleInfo(key: "wi") var wi: Linear
    @ModuleInfo(key: "wo") var wo: Linear
    let hidden: Int
    let activations: [String]

    init(dim: Int, hidden: Int, activations: [String]) {
        precondition(activations.count == 2, "Mlp expects two activation functions")
        self.hidden = hidden
        self.activations = activations
        _wi.wrappedValue = Linear(dim, 2 * hidden, bias: false)
        _wo.wrappedValue = Linear(hidden, dim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let proj = wi(x.asType(wi.weight.dtype))
        let shaped = proj.reshaped(Array(x.shape.dropLast()) + [2, hidden])
        let gate = shaped[.ellipsis, 0, 0...], up = shaped[.ellipsis, 1, 0...]
        let h = dia2Activation(activations[0], gate) * dia2Activation(activations[1], up)
        return wo(h.asType(wo.weight.dtype)).asType(x.dtype)
    }
}

/// `MultiStreamEmbedding` — one embedding table, two projections; the second stream is dropped where it is pad.
public final class Dia2MultiStreamEmbedding: Module {
    @ModuleInfo(key: "embedding") var embedding: Embedding
    @ModuleInfo(key: "main_proj") var mainProj: Linear
    @ModuleInfo(key: "second_proj") var secondProj: Linear
    let padId: Int

    init(vocab: Int, dim: Int, padId: Int, lowRankDim: Int?) {
        let base = lowRankDim ?? dim
        self.padId = padId
        _embedding.wrappedValue = Embedding(embeddingCount: vocab, dimensions: base)
        _mainProj.wrappedValue = Linear(base, dim, bias: false)
        _secondProj.wrappedValue = Linear(base, dim, bias: false)
    }

    /// main, second [B, T] int → [B, T, dim]
    func callAsFunction(_ main: MLXArray, _ second: MLXArray) -> MLXArray {
        let outMain = mainProj(embedding(main).asType(mainProj.weight.dtype))
        let outSecond = secondProj(embedding(second).asType(secondProj.weight.dtype))
        let useSecond = (second .!= Int32(padId)).expandedDimensions(axis: -1)
        return outMain + MLX.where(useSecond, outSecond, MLXArray.zeros(like: outSecond))
    }
}

/// The decoder's `Attention` (forward_incremental): GQA projections, per-head q/k RMSNorm, RoPE at the step position,
/// scale 1.0 SDPA over the cache. T > 1 (the prefix prefill) is the same computation for T frames at once.
public final class Dia2Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Dia2RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Dia2RMSNorm
    let queryHeads: Int, kvHeads: Int, headDim: Int
    let rotary: Dia2Rotary

    init(_ cfg: Dia2Config, dim: Int) {
        let s = cfg.decoder
        queryHeads = s.queryHeads; kvHeads = s.kvHeads; headDim = s.headDim
        _qProj.wrappedValue = Linear(dim, s.queryHeads * s.headDim, bias: false)
        _kProj.wrappedValue = Linear(dim, s.kvHeads * s.headDim, bias: false)
        _vProj.wrappedValue = Linear(dim, s.kvHeads * s.headDim, bias: false)
        _oProj.wrappedValue = Linear(s.queryHeads * s.headDim, dim, bias: false)
        _qNorm.wrappedValue = Dia2RMSNorm(s.headDim, eps: cfg.normEps)
        _kNorm.wrappedValue = Dia2RMSNorm(s.headDim, eps: cfg.normEps)
        rotary = Dia2Rotary(headDim: s.headDim, minTimescale: cfg.ropeMinTimescale, maxTimescale: cfg.ropeMaxTimescale)
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray, cache: Dia2CacheSlot) -> MLXArray {
        let (b, t) = (x.dim(0), x.dim(1))
        let wdt = qProj.weight.dtype
        var q = qProj(x.asType(wdt)).reshaped([b, t, queryHeads, headDim])
        var k = kProj(x.asType(wdt)).reshaped([b, t, kvHeads, headDim])
        let v = vProj(x.asType(wdt)).reshaped([b, t, kvHeads, headDim])
        q = qNorm(q, outDType: x.dtype)
        k = kNorm(k, outDType: x.dtype)
        q = rotary(q, positions: positions)
        k = rotary(k, positions: positions)
        let (kc, vc) = cache.writeAndView(k.transposed(0, 2, 1, 3), v.asType(x.dtype).transposed(0, 2, 1, 3))
        // a multi-frame prefill attends causally (bottom-right aligned: query i sees the cache up to its own frame)
        let y = MLXFast.scaledDotProductAttention(queries: q.transposed(0, 2, 1, 3), keys: kc, values: vc, scale: 1.0,
                                                  mask: t > 1 ? .causal : .none)
        let flat = y.transposed(0, 2, 1, 3).reshaped([b, t, queryHeads * headDim])
        return oProj(flat.asType(oProj.weight.dtype)).asType(x.dtype)
    }
}
