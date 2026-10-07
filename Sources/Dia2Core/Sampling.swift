// Sampling.swift — dia2/runtime/guidance.py, sampler.py and audio/grid.py `mask_audio_logits`, ported.
//
// Upstream's CFG is a FILTER, not a mix: guided = lerp(uncond, cond, scale) picks which vocabulary entries survive (those
// ≥ guided's k-th largest value), and the CONDITIONAL logits of the survivors are what gets sampled. Sampling is
// softmax(logits / temp) → keep the top-k probabilities → renormalise → one multinomial draw; restricting the logits to
// their top-k and drawing categorically is the same distribution. cb0 additionally has the audio pad and bos entries set
// to finfo.min after guidance.
//
// The draw itself goes through a `Dia2Sampler`: `Dia2RandomSampler` for generation (MLXRandom, lazy — the token stays on
// the GPU), the gates' replay sampler to force upstream's recorded picks while comparing every distribution
// (AB-L-0203).

import Foundation
import MLX
import MLXRandom

public enum Dia2DrawKind: Int, Sendable {
    case text = 0, cb0 = 1, dep = 2
}

public struct Dia2Sampling: Sendable {
    public var temperature: Float
    public var topK: Int
    public init(temperature: Float, topK: Int) { self.temperature = temperature; self.topK = topK }
}

public protocol Dia2Sampler: AnyObject {
    /// logits [1, V] float32 (after guidance and masking) → the drawn token, [1] int32 (may stay lazy).
    func draw(_ logits: MLXArray, kind: Dia2DrawKind, stage: Int, sampling: Dia2Sampling) -> MLXArray
    /// The guided logits the CFG filter thresholded for the next draw (nil when CFG is off) — for the gates.
    func observeGuidance(_ guided: MLXArray?, kind: Dia2DrawKind, stage: Int)
}

public extension Dia2Sampler {
    func observeGuidance(_ guided: MLXArray?, kind: Dia2DrawKind, stage: Int) {}
}

public enum Dia2Guidance {
    /// `apply_classifier_guidance` on [2, V] logits (branch 0 conditional, branch 1 unconditional) → [1, V] float32.
    public static func apply(_ logits: MLXArray, scale: Float, filterK: Int) -> MLXArray {
        filter(logits, scale: scale, filterK: filterK).logits
    }

    /// As `apply`, also returning the guided logits the filter thresholded (nil when CFG is off).
    public static func filter(_ logits: MLXArray, scale: Float, filterK: Int) -> (logits: MLXArray, guided: MLXArray?) {
        let cond = logits[0 ..< 1].asType(.float32)
        guard scale != 1.0 else { return (cond, nil) }
        let uncond = logits[1 ..< 2].asType(.float32)
        let guided = uncond + scale * (cond - uncond)                       // torch.lerp(uncond, cond, scale)
        let v = guided.dim(-1)
        guard filterK > 0, v > 0 else { return (cond, guided) }
        let threshold = kthLargest(guided, k: min(filterK, v))
        return (MLX.where(guided .>= threshold, cond, MLXArray(-Float.infinity)), guided)
    }

    /// The k-th largest value along the last axis, keeping dims. Upstream takes `topk(k, sorted=False).values[..., -1:]`,
    /// which on its CPU and CUDA kernels is the k-th largest (the E23 survivor counts confirm it).
    static func kthLargest(_ x: MLXArray, k: Int) -> MLXArray {
        let v = x.dim(-1)
        return partitioned(x, kth: v - k, axis: -1)[.ellipsis, (v - k) ..< (v - k + 1)]
    }

    /// `mask_audio_logits`: the pad and bos entries → finfo(float32).min.
    public static func maskAudio(_ logits: MLXArray, pad: Int, bos: Int) -> MLXArray {
        let v = logits.dim(-1)
        let idx = MLXArray(0 ..< v)
        let hit = (idx .== Int32(pad)) .|| (idx .== Int32(bos))
        return MLX.where(hit, MLXArray(-Float.greatestFiniteMagnitude), logits)
    }

    /// The distribution `sample_token` draws from, as logits: logits / temp restricted to the top-k (others → -inf).
    /// temp ≤ 0 is argmax upstream; callers handle it.
    public static func samplingLogits(_ logits: MLXArray, sampling: Dia2Sampling) -> MLXArray {
        let scaled = logits.asType(.float32) / max(sampling.temperature, 1e-6)
        let v = scaled.dim(-1)
        guard sampling.topK > 0, sampling.topK < v else { return scaled }
        let threshold = kthLargest(scaled, k: sampling.topK)
        return MLX.where(scaled .>= threshold, scaled, MLXArray(-Float.infinity))
    }
}

/// Generation sampler: categorical draws from `samplingLogits`, keyed by a seed (splits one key per draw).
public final class Dia2RandomSampler: Dia2Sampler {
    var key: MLXArray

    public init(seed: UInt64) { key = MLXRandom.key(seed) }

    public func draw(_ logits: MLXArray, kind: Dia2DrawKind, stage: Int, sampling: Dia2Sampling) -> MLXArray {
        if sampling.temperature <= 0 { return argMax(logits, axis: -1).asType(.int32) }
        let parts = MLXRandom.split(key: key)
        key = parts.0
        return MLXRandom.categorical(Dia2Guidance.samplingLogits(logits, sampling: sampling), axis: -1, key: parts.1).asType(.int32)
    }
}
