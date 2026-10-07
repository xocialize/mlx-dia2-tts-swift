// dia2-gates — parity gates against goldens from the upstream nari-labs/dia2 runtime (PyTorch fp32, CPU;
// Tools/oracle-capture/capture_goldens_dia2.py), plus render / validate lanes.
//   swift run -c release dia2-gates --gm | --g1 | --g2 | --g3 | --g5 | --all [--weights DIR] [--goldens DIR]
//   dia2-gates --validate [--quant bf16|fp32] --weights DIR · --cancel --weights BF16_DIR   (GPU; the package, measured)
//   dia2-gates --render-jobs FILE [--system NAME] [--estimate-words] --weights DIR   (GPU; E23 job list → _out/<system>/)

import Dia2Core
import Dia2Mimi
import Foundation
import MLX
import MLXToolKit

var args = Array(CommandLine.arguments.dropFirst())
func flag(_ n: String) -> Bool { if let i = args.firstIndex(of: n) { args.remove(at: i); return true }; return false }
func option(_ n: String) -> String? {
    guard let i = args.firstIndex(of: n), i + 1 < args.count else { return nil }
    let v = args[i + 1]; args.removeSubrange(i ... i + 1); return v
}
let evalDir = URL(fileURLWithPath: ProcessInfo.processInfo.environment["DIA_EVAL"] ?? "/Volumes/Satechi/Development/mlxengine-audio/WIP/dia-eval")
let weightsDir = URL(fileURLWithPath: option("--weights") ?? evalDir.appendingPathComponent("_weights/dia2-2b-fp32").path)
let goldensDir = URL(fileURLWithPath: option("--goldens") ?? evalDir.appendingPathComponent("_goldens/2b").path)

var failures = [String]()
func check(_ ok: Bool, _ what: String) { print((ok ? "  PASS " : "  FAIL ") + what); if !ok { failures.append(what) } }
func npy(_ g: String, _ n: String) throws -> MLXArray { try loadArray(url: goldensDir.appendingPathComponent(g).appendingPathComponent("\(n).npy")) }
func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float { abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self) }
func snr(_ a: MLXArray, _ ref: MLXArray) -> Float {
    let n = sqrt(((a - ref) ** 2).mean()).item(Float.self), s = sqrt((ref ** 2).mean()).item(Float.self)
    return 20 * log10(s / max(n, 1e-12))
}

func gateMimi() throws {
    print("gm Mimi (lifted moshi-swift) vs transformers MimiModel")
    let mimi = try MimiLoader.load(weightsDir.appendingPathComponent("mimi.safetensors"))
    let tokens = try npy("g4_mimi", "tokens").asType(.int32), wave = try npy("g4_mimi", "waveform")
    let y = mimi.decodeFull(tokens.expandedDimensions(axis: 0)).reshaped([-1])
    check(y.dim(0) == wave.dim(0), "gm decode length \(y.dim(0)) == \(wave.dim(0))")
    let n = min(y.dim(0), wave.dim(0))
    let d = maxAbs(y[..<n], wave[..<n])
    check(d < 2e-3, String(format: "gm decode max|Δ| %.2e, SNR %.1f dB", d, snr(y[..<n], wave[..<n])))
    for chunk in [7, 50] {   // the streamed SEANet decode the generator uses; 7 puts boundaries everywhere
        let yc = mimi.decodeChunked(tokens.expandedDimensions(axis: 0), chunk: chunk).reshaped([-1])
        let m = min(yc.dim(0), wave.dim(0))
        check(yc.dim(0) == wave.dim(0) && maxAbs(yc[..<m], wave[..<m]) < 2e-3,
              String(format: "gm chunked decode (chunk %d): length %d, max|Δ| %.2e, SNR %.1f dB, vs whole-take %.2e",
                     chunk, yc.dim(0), maxAbs(yc[..<m], wave[..<m]), snr(yc[..<m], wave[..<m]), maxAbs(yc[..<min(m, n)], y[..<min(m, n)])))
    }
    let ref = try npy("g4_mimi", "ref_audio"), want = try npy("g4_mimi", "ref_codes").asType(.int32)
    let codes = mimi.encodeFull(ref.reshaped([1, 1, -1]))[0].asType(.int32)
    let t = min(codes.dim(1), want.dim(1))
    let same = (codes[0..., ..<t] .== want[0..., ..<t]).asType(.float32).mean().item(Float.self)
    check(codes.shape == want.shape, "gm encode shape \(codes.shape) == \(want.shape)")
    check(same > 0.99, String(format: "gm encode codes agree %.2f %% (all 32 codebooks)", same * 100))
    let first = (codes[0, ..<t] .== want[0, ..<t]).asType(.float32).mean().item(Float.self)
    print(String(format: "    semantic codebook agreement %.2f %%", first * 100))
}

// MARK: - model gates

func loadModel() async throws -> Dia2Model {
    let t0 = Date()
    let m = try await Dia2Model.load(directory: weightsDir)
    print(String(format: "loaded %@ (%@) in %.1f s", weightsDir.lastPathComponent, "\(m.computeDType)", Date().timeIntervalSince(t0)))
    return m
}

func jsonObject(_ name: String) throws -> [String: Any] {
    let data = try Data(contentsOf: goldensDir.appendingPathComponent(name))
    return try JSONSerialization.jsonObject(with: data) as! [String: Any]
}

func entriesMatch(_ got: [Dia2Entry], _ want: [[String: Any]]) -> Bool {
    got.count == want.count && zip(got, want).allSatisfy { g, w in
        g.tokens == (w["tokens"] as! [Int]) && g.text == (w["text"] as! String) && g.padding == (w["padding"] as! Int)
    }
}

/// g1 — swift-transformers on tokenizer.json vs upstream's slow GPT-2 tokenizer; parse_script entries; token ids.
func gateG1(_ m: Dia2Model) throws {
    print("g1 tokenizer + parse_script vs upstream")
    let g = try jsonObject("g1_tokenizer.json")
    let strings = g["strings"] as! [String], ids = g["ids"] as! [[Int]]
    var bad = 0
    for (s, want) in zip(strings, ids) {
        let got = m.tokenizer.encode(s)
        if got != want { bad += 1; print("    \(s.debugDescription): got \(got) want \(want)") }
    }
    check(bad == 0, "g1 encode: \(strings.count - bad)/\(strings.count) strings identical")
    let scripts = g["scripts"] as! [String], ents = g["entries"] as! [[[String: Any]]]
    var badS = 0
    for (s, want) in zip(scripts, ents) {
        let got = Dia2ScriptParser.parse([s], tokenizer: m.tokenizer, ids: m.ids, frameRate: m.frameRate)
        if !entriesMatch(got, want) {
            badS += 1
            print("    \(s.debugDescription):\n      got  \(got.map { ($0.text, $0.tokens, $0.padding) })\n      want \(want.map { ($0["text"]!, $0["tokens"]!, $0["padding"]!) })")
        }
    }
    check(badS == 0, "g1 parse_script: \(scripts.count - badS)/\(scripts.count) scripts' entries identical")
    let c = g["constants"] as! [String: Int], ids2 = m.ids
    let mine = ["card": ids2.card, "new_word": ids2.newWord, "pad": ids2.pad, "bos": ids2.bos, "zero": ids2.zero, "spk1": ids2.spk1,
                "spk2": ids2.spk2, "audio_pad": ids2.audioPad, "audio_bos": ids2.audioBos]
    check(mine == c, "g1 token ids \(mine.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: " "))")
}

/// g2 — the decoder step on upstream's recorded step tokens (both CFG branches), one cache across the generation.
func gateG2(_ m: Dia2Model) throws {
    print("g2 decoder steps vs upstream (recorded tokens, fp32 CPU)")
    let tokens = try npy("g2_steps", "tokens").asType(.int32), positions = try npy("g2_steps", "positions").asType(.int32)
    let action = try npy("g2_steps", "action"), cb0 = try npy("g2_steps", "cb0"), hidden = try npy("g2_steps", "hidden")
    let n = tokens.dim(0)
    let net = m.network.transformer
    let cache = net.makeCache(batch: 2, maxSteps: n + 1, dtype: m.computeDType)
    var worstA: Float = 0, worstC: Float = 0, worstH: Float = 0, worstAt = 0
    let t0 = Date()
    for i in 0 ..< n {
        let (h, a, c) = net.step(tokens[i], position: positions[i].item(Int.self), cache: cache, computeDType: m.computeDType)
        eval(h, a, c)
        let da = maxAbs(a.reshaped([2, -1]), action[i]), dc = maxAbs(c.reshaped([2, -1]), cb0[i])
        if dc > worstC { worstC = dc; worstAt = i }
        worstA = max(worstA, da)
        if i < hidden.dim(0) { worstH = max(worstH, maxAbs(h.reshaped([2, -1]), hidden[i])) }
    }
    print(String(format: "    %d steps in %.1f s", n, Date().timeIntervalSince(t0)))
    check(worstH < 1e-3, String(format: "g2 normed hidden (first %d steps) max|Δ| %.2e", hidden.dim(0), worstH))
    check(worstA < 5e-3, String(format: "g2 action logits max|Δ| %.2e over %d steps × 2 branches", worstA, n))
    check(worstC < 5e-3, String(format: "g2 cb0 logits max|Δ| %.2e (worst at step %d)", worstC, worstAt))
}

/// Forces upstream's recorded picks while comparing every distribution handed to the sampler (AB-L-0203).
final class ReplaySampler: Dia2Sampler {
    let kinds: [Int32], picks: [Int32]
    let text: MLXArray, cb0: MLXArray, dep: MLXArray
    var i = 0, counts = [0, 0, 0]
    var kindMismatch = 0, overrun = 0, survivorMismatch = 0
    var worstAbs: [Float] = [0, 0, 0], worstTV: [Float] = [0, 0, 0]
    var worstTVAt = ""
    /// Draws whose survivor set differs only because upstream's and the port's guided logits straddle a near-tie at
    /// the filter's k-th place (gap ≤ `tieGap`, the order of the logits' own fp32 noise): reported, not gated on TV.
    var ties = [(draw: Int, gap: Float, tv: Float)]()
    let tieGap: Float = 1e-4
    var guided: MLXArray? = nil

    func observeGuidance(_ guided: MLXArray?, kind: Dia2DrawKind, stage: Int) { self.guided = guided }

    /// guided's gap between its k-th and (k+1)-th largest values.
    func boundaryGap(k: Int) -> Float {
        guard let g = guided else { return .infinity }
        let v = sorted(g.reshaped([-1]).asType(.float32)).asArray(Float.self).reversed().map { $0 }
        return k < v.count ? v[k - 1] - v[k] : .infinity
    }

    init(_ g: String) throws {
        kinds = try npy(g, "draw_kinds").asType(.int32).asArray(Int32.self)
        picks = try npy(g, "draw_picks").asType(.int32).asArray(Int32.self)
        text = try npy(g, "text_logits"); cb0 = try npy(g, "cb0_logits"); dep = try npy(g, "dep_logits")
    }

    func draw(_ logits: MLXArray, kind: Dia2DrawKind, stage: Int, sampling: Dia2Sampling) -> MLXArray {
        guard i < kinds.count else { overrun += 1; return argMax(logits, axis: -1).asType(.int32) }
        if Int(kinds[i]) != kind.rawValue { kindMismatch += 1 }
        let k = kind.rawValue
        let ref = [text, cb0, dep][k][counts[k]]
        counts[k] += 1
        let x = logits.reshaped([-1]).asType(.float32)
        let fx = x .> -Float.infinity, fr = ref .> -Float.infinity
        let differ = (fx .!= fr).asType(.int32).sum().item(Int.self)
        let d = MLX.where(fx .&& fr, abs(x - ref), MLXArray(Float(0))).max().item(Float.self)
        let px = softmax(Dia2Guidance.samplingLogits(x, sampling: sampling), axis: -1)
        let pr = softmax(Dia2Guidance.samplingLogits(ref, sampling: sampling), axis: -1)
        let tv = 0.5 * abs(px - pr).sum().item(Float.self)
        worstAbs[k] = max(worstAbs[k], d)
        if differ > 0, case let gap = boundaryGap(k: 50), gap <= tieGap {
            ties.append((i, gap, tv))
        } else {
            survivorMismatch += differ
            if tv > worstTV[k] { worstTV[k] = tv; if tv == worstTV.max()! { worstTVAt = "draw \(i) (\(kind), stage \(stage))" } }
        }
        let pick = picks[i]
        i += 1
        return MLXArray([pick])
    }

    func report(_ tag: String) {
        let names = ["text", "cb0", "dep"]
        check(kindMismatch == 0 && overrun == 0 && i == kinds.count,
              "\(tag) draw sequence: \(i)/\(kinds.count) draws in upstream's order (kind mismatches \(kindMismatch), overrun \(overrun))")
        check(survivorMismatch == 0, "\(tag) guidance survivor sets identical on every draw but boundary near-ties (\(survivorMismatch) entries differ)")
        for t in ties {
            print(String(format: "    near-tie: draw %d — guided gap at the 50th place %.2e ≤ %.0e, one entry swapped; TV there %.2e", t.draw, t.gap, tieGap, t.tv))
        }
        for k in 0 ..< 3 {
            check(worstAbs[k] < 5e-3 && worstTV[k] < 1e-3,
                  String(format: "%@ %@ draws ×%d: logits max|Δ| %.2e, sampling-distribution TV ≤ %.2e", tag, names[k], counts[k], worstAbs[k], worstTV[k]))
        }
        print("    worst TV (outside near-ties) at \(worstTVAt)")
    }
}

func compareResult(_ tag: String, _ r: Dia2Result, golden g: String) throws {
    let want = try npy(g, "tokens").asType(.int32), wave = try npy(g, "waveform"), times = try npy(g, "word_times")
    let meta = try jsonObject("\(g)/meta.json")
    check(r.audioTokens.shape == want.shape, "\(tag) audio grid shape \(r.audioTokens.shape) == \(want.shape)")
    if r.audioTokens.shape == want.shape {
        let same = (r.audioTokens .== want).all().item(Bool.self)
        check(same, "\(tag) undelayed + cropped tokens identical")
    }
    let y = MLXArray(r.waveform)
    check(y.dim(0) == wave.dim(0), "\(tag) waveform length \(y.dim(0)) == \(wave.dim(0))")
    let n = min(y.dim(0), wave.dim(0))
    if n > 0 { check(snr(y[..<n], wave[..<n]) > 60, String(format: "%@ waveform SNR %.1f dB", tag, snr(y[..<n], wave[..<n]))) }
    let words = meta["words"] as! [String]
    let tw = times.asArray(Float.self)
    let sameWords = r.timestamps.map(\.word) == words
    let sameTimes = r.timestamps.count == tw.count && zip(r.timestamps, tw).allSatisfy { abs(Float($0.seconds) - $1) < 1e-4 }
    check(sameWords && sameTimes, "\(tag) word timestamps identical (\(r.timestamps.count) words)")
}

/// g3 — the full generation loop, replayed: state machine, CFG filter, cb0 mask, depformer stages, grid, undelay, Mimi.
func gateG3(_ m: Dia2Model) throws {
    print("g3 generation replay vs upstream (every draw compared)")
    let meta = try jsonObject("g3_replay/meta.json")
    let sampler = try ReplaySampler("g3_replay")
    let t0 = Date()
    let r = try m.generate(meta["script"] as! String, sampler: sampler)
    print(String(format: "    %d frames in %.1f s", r.frames, Date().timeIntervalSince(t0)))
    check(r.frames == meta["n_steps"] as! Int, "g3 loop ran \(r.frames) frames == upstream's \(meta["n_steps"]!)")
    sampler.report("g3")
    try compareResult("g3", r, golden: "g3_replay")
}

/// g5 — two-speaker voice prefix: the plan (entries, new-word steps) from refs.json's words and upstream's Mimi codes,
/// then the prefixed generation replayed.
func gateG5(_ m: Dia2Model) throws {
    print("g5 prefix plan + prefixed replay vs upstream")
    let plan = try jsonObject("g5_prefix_plan.json")
    let refs = try JSONSerialization.jsonObject(with: Data(contentsOf: evalDir.appendingPathComponent("cues/refs.json"))) as! [String: Any]
    func words(_ k: String) -> [Dia2PrefixWord] {
        ((refs[k] as! [String: Any])["words"] as! [[String: Any]]).map {
            Dia2PrefixWord(text: $0["text"] as! String, start: ($0["start"] as! NSNumber).doubleValue, end: ($0["end"] as! NSNumber).doubleValue)
        }
    }
    let aligned = try npy("g5_prefix", "aligned_tokens").asType(.int32)
    let t1 = try npy("g4_mimi", "ref_codes").dim(1)
    let p = m.prefixPlan(s1: (words("ref_s1"), aligned[0..., ..<t1]), s2: (words("ref_s2"), aligned[0..., t1...]))
    check(entriesMatch(p.entries, plan["entries"] as! [[String: Any]]), "g5 prefix entries identical (\(p.entries.count))")
    check(p.newWordSteps == (plan["new_word_steps"] as! [Int]), "g5 new-word steps identical")
    check(p.alignedFrames == plan["aligned_frames"] as! Int, "g5 aligned frames \(p.alignedFrames)")
    let meta = try jsonObject("g5_prefix/meta.json")
    let sampler = try ReplaySampler("g5_prefix")
    let t0 = Date()
    let r = try m.generate(meta["script"] as! String, prefix: p, sampler: sampler)
    print(String(format: "    warmup %d + %d frames in %.1f s", p.alignedFrames, r.frames, Date().timeIntervalSince(t0)))
    check(r.frames == meta["n_steps"] as! Int, "g5 loop ran \(r.frames) frames == upstream's \(meta["n_steps"]!)")
    sampler.report("g5")
    try compareResult("g5", r, golden: "g5_prefix")
}

do {
    let all = flag("--all")
    let wantG1 = all || flag("--g1"), wantG2 = all || flag("--g2"), wantG3 = all || flag("--g3"), wantG5 = all || flag("--g5")
    let wantGM = all || flag("--gm")
    try Device.withDefaultDevice(.cpu) {
        if wantGM { try gateMimi() }
    }
    let validateQuant = option("--quant").flatMap(Quant.init(rawValue:)) ?? .bf16
    if flag("--validate") { try await gateValidate(quant: validateQuant) }
    if flag("--cancel") { try await gateCancel() }
    if let jobs = option("--render-jobs") {
        let m = try await loadModel()
        let system = option("--system") ?? "swift-\(weightsDir.lastPathComponent)"
        try renderJobs(m, jobsFile: URL(fileURLWithPath: jobs), system: system, cacheLimitMB: Int(option("--cache-limit-mb") ?? "2048")!,
                       estimateWords: flag("--estimate-words"))
    }
    if wantG1 || wantG2 || wantG3 || wantG5 {
        let m = try await loadModel()
        try Device.withDefaultDevice(.cpu) {
            if wantG1 { try gateG1(m) }
            if wantG2 { try gateG2(m) }
            if wantG3 { try gateG3(m) }
            if wantG5 { try gateG5(m) }
        }
    }
    print(failures.isEmpty ? "ALL PASS" : "FAILED: \(failures.count) — \(failures)")
    exit(failures.isEmpty ? 0 : 1)
} catch {
    print("error: \(error)"); exit(1)
}
