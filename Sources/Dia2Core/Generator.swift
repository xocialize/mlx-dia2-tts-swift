// Generator.swift — dia2/runtime/generator.py (build_initial_state, warmup_with_prefix, run_generation_loop),
// dia2/runtime/voice_clone.py (words_to_entries, build_prefix_plan) and Dia2.generate (engine.py), ported 1:1,
// upstream quirks included:
//
//   • two CFG branches — branch 0 reads the script, branch 1 always reads (zero, pad) on the text streams;
//   • the generation loop starts AT aligned_frames − 1 after a prefix warmup, so that position is decoded twice
//     (the decoder cache appends, as upstream's);
//   • `first_word_frame` keys on main == new_word, which the second-stream multiplexer never emits — so the crop is
//     the start step (0, or aligned_frames − 1 after a prefix);
//   • speaker 2's prefix new-word steps are offset by speaker 1's frame count but not by the +3 speaker 1 gets.
//
// The delayed audio grid is kept as one [32] int32 column per frame (lazy MLXArrays — the depformer's draws stay on
// the GPU); one CPU sync per frame, on the action token the state machine needs.

import Dia2Mimi
import Foundation
import MLX

public struct Dia2GenerationConfig: Sendable {
    public var text = Dia2Sampling(temperature: 0.6, topK: 50)
    public var audio = Dia2Sampling(temperature: 0.8, topK: 50)
    public var cfgScale: Float = 2.0
    public var cfgFilterK = 50
    public var initialPadding = 2
    /// Keep the prefix audio in the output (upstream `include_prefix`).
    public var includePrefixAudio = false
    /// Overrides runtime.max_context_steps (1500 frames = 120 s) — the loop's step budget.
    public var maxContextSteps: Int? = nil

    public init() {}
}

/// A word of a voice prefix with its timing in the clip (seconds) — upstream gets these from whisper_timestamped;
/// here the caller supplies them (the fleet aligner).
public struct Dia2PrefixWord: Sendable, Equatable {
    public var text: String
    public var start: Double
    public var end: Double
    public init(text: String, start: Double, end: Double) { self.text = text; self.start = start; self.end = end }
}

/// One speaker's voice prefix: the clip (mono, 24 kHz) and its word timings.
public struct Dia2Voice: Sendable {
    public var audio: [Float]
    public var words: [Dia2PrefixWord]
    public init(audio: [Float], words: [Dia2PrefixWord]) { self.audio = audio; self.words = words }
}

public struct Dia2PrefixPlan {
    public var entries: [Dia2Entry]
    public var newWordSteps: [Int]
    public var alignedTokens: MLXArray      // [32, T] int32 (undelayed Mimi codes)
    public var alignedFrames: Int { alignedTokens.dim(1) }
}

public struct Dia2Result {
    public let audioTokens: MLXArray        // [32, T] int32, undelayed, cropped
    public let waveform: [Float]            // 24 kHz mono, clamped to [-1, 1]
    public let sampleRate: Int
    public let timestamps: [(word: String, seconds: Double)]
    public let frames: Int                  // generation-loop steps run
    /// The script's last word was consumed (end_step reached) — false when the step budget ran out first and the take
    /// is truncated (upstream returns it silently).
    public let finished: Bool
}

extension Dia2Model {
    // MARK: prefix

    /// `words_to_entries`.
    public func prefixEntries(_ words: [Dia2PrefixWord], speakerToken: Int) -> ([Dia2Entry], [Int]) {
        var entries = [Dia2Entry](), steps = [Int]()
        guard !words.isEmpty else { return (entries, steps) }
        var pendingPrefix: String? = speakerToken == ids.spk1 ? "[S1]" : (speakerToken == ids.spk2 ? "[S2]" : nil)
        var currentPos = 0
        func frame(_ s: Double) -> Int { Int((s * frameRate).rounded(.toNearestOrEven)) }
        for (idx, word) in words.enumerated() {
            let tokens = pendingPrefix.map { tokenizer.encode("\($0) \(word.text)") } ?? tokenizer.encode(word.text)
            pendingPrefix = nil
            let startFrame = max(currentPos + 1, frame(word.start))
            let endFrame = startFrame + tokens.count
            steps.append(startFrame - 1)
            let nextWordStart = idx < words.count - 1
                ? max(endFrame + 1, frame(words[idx + 1].start))
                : max(endFrame + 1, frame(words[words.count - 1].end))
            entries.append(Dia2Entry(tokens: tokens, text: word.text, padding: max(0, nextWordStart - startFrame - 1)))
            currentPos = endFrame
        }
        return (entries, steps)
    }

    /// Mimi codes of a mono 24 kHz clip → [32, T] int32 (`encode_audio_tokens`).
    public func encodeAudio(_ audio: [Float]) -> MLXArray {
        let codes = mimi.encodeFull(MLXArray(audio).reshaped([1, 1, -1]))[0].asType(.int32)
        eval(codes)
        return codes
    }

    /// `build_prefix_plan` from precomputed pieces (the gates feed upstream's own codes and words through here).
    public func prefixPlan(s1: (words: [Dia2PrefixWord], codes: MLXArray), s2: (words: [Dia2PrefixWord], codes: MLXArray)?) -> Dia2PrefixPlan {
        let (e1, steps1) = prefixEntries(s1.words, speakerToken: ids.spk1)
        var entries = e1
        var newWordSteps = steps1.map { $0 + 3 }           // "Match legacy BOS/PAD offset"
        var tokens = s1.codes
        if let s2 {
            let (e2, steps2) = prefixEntries(s2.words, speakerToken: ids.spk2)
            let spk1Frames = tokens.dim(1)
            newWordSteps.append(contentsOf: steps2.map { $0 + spk1Frames })
            entries.append(contentsOf: e2)
            tokens = concatenated([tokens, s2.codes], axis: 1)
        }
        return Dia2PrefixPlan(entries: entries, newWordSteps: newWordSteps, alignedTokens: tokens)
    }

    public func prefixPlan(s1: Dia2Voice, s2: Dia2Voice?) -> Dia2PrefixPlan {
        prefixPlan(s1: (s1.words, encodeAudio(s1.audio)), s2: s2.map { ($0.words, encodeAudio($0.audio)) })
    }

    // MARK: generate

    /// `Dia2.generate`: script (with [S1] / [S2] turns) → waveform + word timestamps.
    public func generate(_ script: String, config gen: Dia2GenerationConfig = Dia2GenerationConfig(),
                         prefix: Dia2PrefixPlan? = nil, sampler: Dia2Sampler,
                         progress: ((Int) -> Void)? = nil) throws -> Dia2Result {
        var entries = prefix?.entries ?? []
        entries.append(contentsOf: parse(script))
        let machine = stateMachine(initialPadding: gen.initialPadding)
        let state = machine.newState(entries)
        let run = Dia2Run(model: self, machine: machine, prefix: prefix, maxContext: gen.maxContextSteps ?? config.maxContextSteps)
        var startStep = 0
        if let prefix { startStep = try run.warmup(prefix, state: state) }
        let firstWordFrame = try run.loop(state: state, gen: gen, startStep: startStep, sampler: sampler, progress: progress)

        // undelay, crop, decode
        var aligned = run.undelayed()
        let includePrefix = prefix != nil && gen.includePrefixAudio
        var crop = includePrefix ? 0 : max(firstWordFrame, 0)
        if crop > 0 && crop < aligned.dim(1) {
            aligned = aligned[0..., crop...]
        } else if crop >= aligned.dim(1) {
            crop = 0
        }
        var waveform = [Float]()
        if aligned.dim(1) > 0 {
            let pcm = clip(mimi.decodeChunked(aligned.expandedDimensions(axis: 0)), min: -1.0, max: 1.0).reshaped([-1])
            waveform = pcm.asType(.float32).asArray(Float.self)
        }
        var transcript = state.transcript
        if let prefix, !includePrefix {
            transcript = transcript.count > prefix.entries.count ? Array(transcript.dropFirst(prefix.entries.count)) : []
        }
        let timestamps = transcript.compactMap { (word, step) -> (word: String, seconds: Double)? in
            let adj = step - crop
            return adj < 0 ? nil : (word, Double(adj) / max(frameRate, 1.0))
        }
        return Dia2Result(audioTokens: aligned, waveform: waveform, sampleRate: sampleRate, timestamps: timestamps, frames: run.framesRun,
                          finished: state.endStep != nil)
    }
}

/// One generation's mutable state (`GenerationState` + the loop's locals).
final class Dia2Run {
    let model: Dia2Model
    let machine: Dia2StateMachine
    let ids: Dia2TokenIds
    let delays: [Int]
    let delayMask: [MLXArray]               // per step t < maxDelay: [32] bool, delay > t
    let maxContext: Int
    let totalSteps: Int
    var audioBuf: [MLXArray?]               // delayed grid, one [32] int32 column per frame; nil = ungenerated
    var text0: (Int, Int)                   // branch 0's (main, second) for the next step
    let decCache: Dia2KVCache
    let depCache: Dia2KVCache
    var lastStep = -1
    var framesRun = 0

    init(model: Dia2Model, machine: Dia2StateMachine, prefix: Dia2PrefixPlan?, maxContext: Int) {
        self.model = model
        self.machine = machine
        ids = model.ids
        delays = model.delays
        let maxDelay = delays.max() ?? 0
        let delayArr = MLXArray(delays.map(Int32.init))
        delayMask = (0 ..< maxDelay).map { t in delayArr .> Int32(t) }
        self.maxContext = maxContext
        var prefixLen = 0
        var delayed: MLXArray? = nil
        if let prefix {
            delayed = Dia2Run.delayFrames(prefix.alignedTokens, delays: delays, pad: ids.audioPad)
            prefixLen = delayed!.dim(1)
        }
        totalSteps = max(maxContext + prefixLen + 1, maxContext)
        audioBuf = Array(repeating: nil, count: totalSteps)
        if let delayed {
            for t in 0 ..< delayed.dim(1) { audioBuf[t] = delayed[0..., t] }
        }
        text0 = (ids.bos, ids.pad)
        let dt = model.computeDType
        decCache = model.network.transformer.makeCache(batch: 2, maxSteps: totalSteps, dtype: dt)
        depCache = model.network.depformer.makeCache(batch: 2, dtype: dt)
    }

    /// `delay_frames`: [C, T] → [C, T + maxDelay], channel c shifted right by delay[c], pad elsewhere.
    static func delayFrames(_ aligned: MLXArray, delays: [Int], pad: Int) -> MLXArray {
        let c = aligned.dim(0)
        let maxDelay = delays.max() ?? 0
        let rows = (0 ..< c).map { idx -> MLXArray in
            let d = delays[idx]
            let left = MLXArray.full([d], values: MLXArray(Int32(pad)))
            let right = MLXArray.full([maxDelay - d], values: MLXArray(Int32(pad)))
            return concatenated([left, aligned[idx].asType(.int32), right], axis: 0)
        }
        return stacked(rows, axis: 0)
    }

    /// The step's [2, 34] input: text streams per branch, then the 32 audio channels (bos where delay > t).
    func stepTokens(_ t: Int, audio: MLXArray) -> MLXArray {
        let text = MLXArray([Int32(text0.0), Int32(text0.1), Int32(ids.zero), Int32(ids.pad)], [2, 2])
        let row = broadcast(audio.reshaped([1, -1]), to: [2, audio.dim(0)])
        return concatenated([text, row], axis: 1)
    }

    /// `_fill_audio_channels`.
    func audioColumn(_ t: Int) throws -> MLXArray {
        let bos = MLXArray(Int32(ids.audioBos))
        guard t < audioBuf.count else { return MLXArray.full([delays.count], values: bos) }
        let masked = t < delayMask.count ? delayMask[t] : nil
        guard let col = audioBuf[t] else {
            // ungenerated: upstream would embed id -2 (an error) unless every channel is still inside its delay
            guard delays.allSatisfy({ $0 > t }) else { throw Dia2Error.invalidInput("audio grid column \(t) read before it was generated") }
            return MLXArray.full([delays.count], values: bos)
        }
        return masked.map { MLX.where($0, bos, col) } ?? col
    }

    /// `warmup_with_prefix` → the loop's start step. Upstream steps the decoder frame by frame and discards its
    /// outputs; the step inputs depend only on the prefix codes and the FORCED state machine, so they are built first
    /// and the decoder fills its cache in causal chunks (`prefillChunk` frames per pass).
    func warmup(_ plan: Dia2PrefixPlan, state: Dia2State, prefillChunk: Int = 256) throws -> Int {
        let net = model.network.transformer
        let bos = MLXArray(Int32(ids.audioBos))
        let tokens = plan.alignedTokens.asType(.int32)
        let newWordSteps = Set(plan.newWordSteps)
        var inputs = [MLXArray]()
        for t in 0 ..< plan.alignedFrames {
            let col = stacked(delays.enumerated().map { cb, d in t - d >= 0 ? tokens[cb, t - d] : bos }, axis: 0)
            inputs.append(stepTokens(t, audio: col))
            let forced = newWordSteps.contains(t) ? ids.newWord : ids.pad
            let (main, aux, _) = machine.process(step: t, state: state, token: forced, isForced: true)
            text0 = (main, aux == -1 ? ids.pad : aux)
        }
        for start in stride(from: 0, to: inputs.count, by: prefillChunk) {
            try Task.checkCancellation()
            let chunk = stacked(Array(inputs[start ..< min(start + prefillChunk, inputs.count)]), axis: 1)    // [2, n, 34]
            _ = net.forward(chunk, startPosition: start, cache: decCache, computeDType: model.computeDType)
            eval(decCache.slots.flatMap { [$0.keys, $0.values] })
        }
        return max(plan.alignedFrames - 1, 0)
    }

    /// `run_generation_loop` → first_word_frame.
    func loop(state: Dia2State, gen: Dia2GenerationConfig, startStep: Int, sampler: Dia2Sampler,
              progress: ((Int) -> Void)?) throws -> Int {
        let net = model.network
        let dt = model.computeDType
        let maxDelay = delays.max() ?? 0
        let flushTail = maxDelay + machine.maxPadding
        var firstWordFrame: Int? = nil
        var eosCutoff: Int? = nil
        lastStep = startStep - 1
        for offset in 0 ..< maxContext {
            let t = startStep + offset
            if let eosCutoff, t >= eosCutoff { break }
            if t + 1 >= audioBuf.count { break }
            try Task.checkCancellation()
            depCache.reset()
            let tokens = stepTokens(t, audio: try audioColumn(t))
            let (hidden, action, cb0) = net.transformer.step(tokens, position: t, cache: decCache, computeDType: dt)

            let guidedText = Dia2Guidance.filter(action[0..., 0, 0...], scale: gen.cfgScale, filterK: gen.cfgFilterK)
            sampler.observeGuidance(guidedText.guided, kind: .text, stage: 0)
            let textToken = sampler.draw(guidedText.logits, kind: .text, stage: 0, sampling: gen.text).item(Int.self)
            let (main, aux, _) = machine.process(step: t, state: state, token: textToken)
            let second = aux == -1 ? ids.pad : aux
            if firstWordFrame == nil && main == ids.newWord { firstWordFrame = t - gen.initialPadding }
            text0 = (main, second)

            let filteredCb0 = Dia2Guidance.filter(cb0[0..., 0, 0...], scale: gen.cfgScale, filterK: gen.cfgFilterK)
            sampler.observeGuidance(filteredCb0.guided, kind: .cb0, stage: 0)
            let guidedCb0 = Dia2Guidance.maskAudio(filteredCb0.logits, pad: ids.audioPad, bos: ids.audioBos)
            var picks = [sampler.draw(guidedCb0, kind: .cb0, stage: 0, sampling: gen.audio).reshaped([1])]
            var prev = broadcast(picks[0], to: [2])
            for stage in 0 ..< net.depformer.depth {
                let logits = net.depformer.stage(stage, prevAudio: prev, hidden: hidden, cache: depCache, computeDType: dt)
                let filtered = Dia2Guidance.filter(logits, scale: gen.cfgScale, filterK: gen.cfgFilterK)
                sampler.observeGuidance(filtered.guided, kind: .dep, stage: stage)
                let pick = sampler.draw(filtered.logits, kind: .dep, stage: stage, sampling: gen.audio).reshaped([1])
                picks.append(pick)
                prev = broadcast(pick, to: [2])
            }
            let column = concatenated(picks, axis: 0)
            asyncEval(column)
            audioBuf[t + 1] = column
            lastStep = t
            if eosCutoff == nil, let end = state.endStep { eosCutoff = end + flushTail }
            framesRun = offset + 1
            progress?(framesRun)
        }
        let first = firstWordFrame ?? startStep
        let limit = lastStep < startStep ? min(startStep + 1, audioBuf.count) : min(lastStep + 2, audioBuf.count)
        let pad = MLXArray.full([delays.count], values: MLXArray(Int32(ids.audioPad)))
        audioBuf = Array(audioBuf[0 ..< limit].map { $0 ?? pad })        // trim_audio: ungenerated → pad
        return first
    }

    /// The trimmed delayed grid, [32, limit] int32.
    func delayedGrid() -> MLXArray {
        guard !audioBuf.isEmpty else { return MLXArray.zeros([delays.count, 0], dtype: .int32) }
        return stacked(audioBuf.map { $0! }, axis: 1)
    }

    /// `undelay_frames`.
    func undelayed() -> MLXArray {
        let grid = delayedGrid()
        let maxDelay = delays.max() ?? 0
        let target = max(0, grid.dim(1) - maxDelay)
        guard target > 0 else { return MLXArray.zeros([delays.count, 0], dtype: .int32) }
        let out = stacked(delays.enumerated().map { cb, d in grid[cb, d ..< (d + target)] }, axis: 0)
        eval(out)
        return out
    }
}
