import AVFoundation
import Dia2Core
import Foundation
import MLX
import MLXNN
import MLXToolKit

/// Dia2-2B (Nari Labs, Apache-2.0) on the canonical `tts` surface — the fleet's two-speaker SCENE renderer: a script
/// with `[S1]` / `[S2]` turns → ONE 24 kHz mono `.wav` in which the two speakers trade turns with conversational gaps.
/// Text without tags is speaker 1 alone.
///
/// Engine-owned lifecycle (C13): the engine constructs from a `Dia2TTSConfiguration`, materializes the declared source
/// into its store, pages weights in with `load()`, drives `run(_:)`, and reclaims with `unload()`.
///
/// Voice plane — Dia2 conditions each speaker on a voice PREFIX (a clip + its words):
/// - `.auto` — no prefix: a fresh voice pair, fixed by `metaData.seed`.
/// - `.referenceAudio(clip)` + `referenceTranscript` — speaker 1's prefix. Required: the transcript (the prefix's
///   words are fed to the script stream). Word timings come from `metaData.referenceWords` when the caller has them
///   (an aligner's output), else they are estimated from the transcript (`Dia2WordTiming`).
/// - Speaker 2's prefix (needs speaker 1's): INTERIM `metaData` keys until the contract carries a second voice —
///   `speaker2Audio` (base64 `.wav`), `speaker2Transcript`, optional `speaker2Words`.
/// - `.named` — rejected: Dia2 has no preset voices.
/// A single prefix conditions Dia2 only weakly (E23: 0.44 cosine to the reference); with BOTH speakers prefixed it
/// holds (0.80 — the shape for scenes and for speaking against the other party's audio). For single-voice cloning
/// the roster's cloners are stronger.
///
/// `metaData` keys (package-specific, C5):
/// - `seed` (int): reproducible sampling within Swift (MLXRandom; NOT the PyTorch reference's stream).
/// - `cfgScale` (double, 2.0), `textTemperature` (double, 0.6), `audioTemperature` (double, 0.8), `topK` (int, 50):
///   upstream's GenerationConfig.
/// - `referenceWords` / `speaker2Words` (array of {text, start, end} seconds): prefix word timings.
/// - `speaker2Audio` (string, base64 .wav), `speaker2Transcript` (string): speaker 2's prefix (interim).
/// - `includePrefixAudio` (bool, false): keep the prefix clips at the head of the output.
///
/// One take covers at most 1 500 frames (120 s); a script that does not finish within it is refused, not truncated.
@InferenceActor
public final class Dia2TTSPackage: ModelPackage {
    public typealias Configuration = Dia2TTSConfiguration

    /// Split footprint of the published bf16 tier, MEASURED as phys_footprint through this package (`dia2-gates
    /// --validate`, quiet box, MLX pool at the engine's 2 GiB automatic cap, 2026-10-06; MEASUREMENTS.md), pool-inclusive
    /// as the fleet declares (AB-L-0113 / AB-L-0155): phys 4.41 GB after load; 7.01 GB at the highest reading over takes
    /// of 3–101 s, both speakers prefixed → 2.60 GB activation. Of that, Dia2's own MLX working set is ≤ 1.21 GB, flat
    /// in take length (Mimi's SEANet decoder streams — `decodeChunked`; the decoder's KV cache is preallocated for the
    /// full 1 500 frames); the rest is the engine's recycling pool. (The unpublished fp32 parity tier measured phys
    /// 8.25 GB + 2.71 GB; it is not a declared, engine-selectable tier.)
    nonisolated static let bf16ResidentBytes: UInt64 = 4_500_000_000
    nonisolated static let bf16PeakActivationBytes: UInt64 = 3_000_000_000

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // C7: nari-labs/Dia2-2B is Apache-2.0 (the transformer, depformer and tokenizer files); the bundled Mimi
            // codec weights are kyutai/mimi, CC-BY-4.0 (attribution in THIRD_PARTY_NOTICES). C8: port code MIT, the
            // lifted moshi-swift Mimi MIT (Kyutai).
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "nari-labs/Dia2-2B", revision: "7abae125471a73b0fc6b9d413cb15f4ae1e771d8", tier: 1),
            requirements: RequirementsManifest(
                footprints: [
                    QuantFootprint(quant: .bf16, residentBytes: bf16ResidentBytes, peakActivationBytes: bf16PeakActivationBytes),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                TTSContract.descriptor(
                    name: "dia2-2b",
                    summary: "Dia2-2B (Nari Labs) two-speaker dialogue TTS (.wav, 24 kHz mono, English): a script with "
                        + "[S1] / [S2] turns renders as ONE take with natural turn-taking — background conversations, "
                        + "generated dialogue. voice.auto = a fresh voice pair per seed; voice.referenceAudio + "
                        + "referenceTranscript = speaker 1's voice prefix (weak alone; strong with speaker 2's prefix "
                        + "too, metaData speaker2Audio / speaker2Transcript). Nonverbal tags like (laughs) (sighs) are "
                        + "accepted but rarely performed. ≤ 120 s per take. metaData: seed / cfgScale / "
                        + "textTemperature / audioTemperature / topK / referenceWords / speaker2Words / includePrefixAudio.",
                    modes: [.expressive]
                )
            ]
        )
    }

    private let configuration: Configuration
    private var model: Dia2Model?
    /// The last prefix plan, keyed by its inputs (the Mimi encode of a clip is a pure function of its bytes) — a Dub
    /// scene re-renders against the same two voices, seed after seed.
    private var cachedPlan: (key: String, plan: Dia2PrefixPlan)?

    /// C14/INF seam: the module graphs this package holds. RMSNorm transformers only (no BatchNorm / Dropout).
    var inferenceModeGraphs: [String: MLXNN.Module?] {
        ["dia2": model?.network, "mimi": model?.mimi]
    }

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    // MARK: - Lifecycle

    public func load() async throws {
        guard model == nil else { return }
        // Materialization is ENGINE-EXECUTED (contract 1.24): this guard is the offline backstop only.
        let storeRoot = configuration.modelsRootDirectory
        let missing = configuration.missingWeightSources(storeRoot: storeRoot)
        guard missing.isEmpty else {
            throw Dia2PackageError.missingWeights(
                "sources not materialized: \(missing.map(\.role).joined(separator: ", ")) "
                + (storeRoot.map { "(store: \($0.path))" } ?? "(no models root set)"))
        }
        try Task.checkCancellation()
        guard let dir = configuration.resolved(storeRoot: storeRoot).modelDirectory else {
            throw Dia2PackageError.missingWeights("unresolved weight directory (no store root)")
        }
        model = try await Dia2Model.load(directory: dir, dtype: configuration.quant == .fp32 ? .float32 : .bfloat16)
    }

    public func unload() async {
        model = nil
        cachedPlan = nil
        MLX.Memory.clearCache()   // release the retained MLX pool so eviction frees RSS
    }

    // MARK: - Run

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run(). Mid-run cadence: every prefix warmup frame and every
        // generated frame (Dia2Core checks Task cancellation); the CancellationError is rethrown UNCHANGED.
        try Task.checkCancellation()
        guard let model else { throw PackageError.notLoaded }
        guard request.capability == .tts, let tts = request as? TTSRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        let script = tts.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !script.isEmpty else { throw PackageError.unsupportedRequestFeature("empty text") }
        let prefix = try prefixPlan(model: model, request: tts)
        try Task.checkCancellation()

        var gen = Dia2GenerationConfig()
        let meta = tts.metaData
        if let v = meta.doubleValue("cfgScale") { gen.cfgScale = Float(v) }
        if let v = meta.doubleValue("textTemperature") { gen.text.temperature = Float(v) }
        if let v = meta.doubleValue("audioTemperature") { gen.audio.temperature = Float(v) }
        if let k = meta.intValue("topK") { gen.text.topK = k; gen.audio.topK = k }
        if case .bool(let keep)? = meta["includePrefixAudio"] { gen.includePrefixAudio = keep }
        guard gen.cfgScale > 0, gen.text.topK > 0 else {
            throw PackageError.unsupportedRequestFeature("cfgScale and topK must be positive")
        }
        let seed = meta.intValue("seed").map { UInt64(bitPattern: Int64($0)) } ?? UInt64.random(in: 0 ... UInt64.max)
        let result = try model.generate(script, config: gen, prefix: prefix, sampler: Dia2RandomSampler(seed: seed),
                                        progress: { RunProgress.report(.generate, step: $0) })
        try Task.checkCancellation()
        guard result.finished else {
            throw PackageError.unsupportedRequestFeature(
                "script longer than one Dia2 take (\(model.config.maxContextSteps) frames = "
                + "\(Int(Double(model.config.maxContextSteps) / model.frameRate)) s) — split the scene")
        }
        let wav = Dia2AudioIO.encodeWAV16(samples: result.waveform, sampleRate: result.sampleRate)
        return TTSResponse(audio: Audio(format: .wav, data: wav, sampleRate: result.sampleRate, channels: 1))
    }

    /// The voice prefix the request asks for (nil = none).
    func prefixPlan(model: Dia2Model, request: TTSRequest) throws -> Dia2PrefixPlan? {
        let meta = request.metaData
        switch request.voice.selection {
        case .auto:
            if meta["speaker2Audio"] != nil {
                throw PackageError.unsupportedRequestFeature(
                    "metaData.speaker2Audio needs speaker 1's prefix too (voice.referenceAudio + referenceTranscript)")
            }
            return nil
        case .named(let id):
            throw PackageError.unsupportedRequestFeature(
                "voice.named(\"\(id)\") — Dia2 has no preset voices; use voice.auto (a seeded voice pair) or "
                + "voice.referenceAudio + referenceTranscript (a voice prefix)")
        case .referenceAudio(let clip):
            guard let transcript = request.referenceTranscript?.trimmingCharacters(in: .whitespacesAndNewlines), !transcript.isEmpty else {
                throw PackageError.unsupportedRequestFeature(
                    "voice.referenceAudio needs referenceTranscript — Dia2 reads the prefix's words on its script stream")
            }
            var s2: (Audio, String, MetaValue?)? = nil
            if case .string(let b64)? = meta["speaker2Audio"] {
                guard let data = Data(base64Encoded: b64) else {
                    throw PackageError.unsupportedRequestFeature("metaData.speaker2Audio is not base64")
                }
                guard case .string(let t)? = meta["speaker2Transcript"], !t.trimmingCharacters(in: .whitespaces).isEmpty else {
                    throw PackageError.unsupportedRequestFeature("metaData.speaker2Audio needs metaData.speaker2Transcript")
                }
                s2 = (Audio(format: .wav, data: data, sampleRate: nil, channels: nil), t, meta["speaker2Words"])
            }
            let key = [Dia2AudioIO.digest(clip.data), transcript, String(describing: meta["referenceWords"]),
                       s2.map { Dia2AudioIO.digest($0.0.data) + $0.1 + String(describing: $0.2) } ?? "-"].joined(separator: "|")
            if let cachedPlan, cachedPlan.key == key { return cachedPlan.plan }
            let v1 = try voice(clip, transcript: transcript, words: meta["referenceWords"], field: "referenceWords")
            let v2 = try s2.map { try voice($0.0, transcript: $0.1, words: $0.2, field: "speaker2Words") }
            let plan = model.prefixPlan(s1: v1, s2: v2)
            cachedPlan = (key, plan)
            return plan
        }
    }

    func voice(_ clip: Audio, transcript: String, words: MetaValue?, field: String) throws -> Dia2Voice {
        let samples = try Dia2AudioIO.decode24k(clip)
        guard samples.count >= 24_000 / 2 else { throw PackageError.unsupportedRequestFeature("voice prefix clip shorter than 0.5 s") }
        if let words {
            return Dia2Voice(audio: samples, words: try Self.parseWords(words, field: field))
        }
        return Dia2Voice(audio: samples, words: Dia2WordTiming.estimate(transcript: transcript, audio: samples, sampleRate: 24_000))
    }

    nonisolated static func parseWords(_ value: MetaValue, field: String) throws -> [Dia2PrefixWord] {
        guard case .array(let items) = value else { throw PackageError.unsupportedRequestFeature("metaData.\(field) must be an array") }
        return try items.map { item in
            guard case .object(let o) = item, case .string(let text)? = o["text"],
                  let start = o["start"].flatMap(number), let end = o["end"].flatMap(number) else {
                throw PackageError.unsupportedRequestFeature("metaData.\(field) entries are {text, start, end}")
            }
            return Dia2PrefixWord(text: text, start: start, end: end)
        }
    }

    nonisolated static func number(_ v: MetaValue) -> Double? {
        switch v {
        case .double(let d): return d
        case .int(let i): return Double(i)
        default: return nil
        }
    }
}

extension Dia2TTSPackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration { .of(Dia2TTSPackage.self) }
}

/// Wrapper-level errors (weight resolution). Runtime request errors use `PackageError`.
public enum Dia2PackageError: Error, CustomStringConvertible {
    case missingWeights(String)
    public var description: String {
        switch self {
        case .missingWeights(let why): return "Dia2 weights unavailable: \(why)"
        }
    }
}

extension MetaData {
    func intValue(_ key: String) -> Int? {
        if case .int(let value)? = self[key] { return value }
        return nil
    }

    func doubleValue(_ key: String) -> Double? {
        switch self[key] {
        case .double(let value)?: return value
        case .int(let value)?: return Double(value)
        default: return nil
        }
    }
}
