// ConformanceTests.swift — Dia2 through the engine's offline gates.
//
// These run without weights and without a kernel: manifest/declaration checks (C0–C13), the materialization gate
// (MAT-1..5), the cancellation gate (CAN-1..3), the request plane, and the pure-Swift runtime pieces (state machine,
// word-timing estimate, CFG filter on toy logits). The live checks — real load, real run, measured footprint — are
// the `dia2-gates` CLI lanes, because the SPM test product's metallib is unreliable for GPU work.

import Foundation
import MLX
import MLXToolKit
import MLXServeConformance
import MLXServeCore
import XCTest

@testable import Dia2Core
@testable import MLXDia2TTS

final class ManifestConformanceTests: XCTestCase {

    /// C7/C8 (1.49.0) — the bundle declares BOTH weight licences (Dia2 Apache-2.0 + Mimi CC-BY-4.0) and MIT port
    /// code; the default policy judges each and admits.
    func testLicenseDeclaresBothWeightSetsAndAdmits() {
        let license = Dia2TTSPackage.manifest.license
        XCTAssertEqual(license.weightLicense, .apache2)
        XCTAssertEqual(license.additionalWeightLicenses, [.ccBy4])
        XCTAssertEqual(license.weightLicenses, [.apache2, .ccBy4])
        XCTAssertEqual(license.portCodeLicense, .mit)
        XCTAssertTrue(LicensePolicy.permissiveOnly.evaluate(license).isAdmitted)
    }

    /// C-memory — split footprint for the one published tier, at least what the validate lane measured in
    /// phys_footprint (pool-inclusive). The fp32 parity tier is unpublished and undeclared.
    func testFootprintIsTheMeasuredBf16Tier() {
        let footprints = Dia2TTSPackage.manifest.requirements.footprints
        XCTAssertEqual(footprints.map(\.quant), [.bf16])
        XCTAssertGreaterThanOrEqual(footprints[0].residentBytes, 4_410_000_000)        // measured phys after load
        XCTAssertGreaterThanOrEqual(footprints[0].peakActivationBytes, 2_600_000_000)  // measured phys peak − resident
    }

    func testSpecialtiesAreRegistered() {
        for weight in Dia2TTSPackage.manifest.specialties {
            XCTAssertTrue(weight.specialty.isRegistered, "unregistered specialty \(weight.specialty.rawValue)")
        }
    }

    /// C1 — the package serves `tts` and nothing else. It declares a two-speaker cast (1.49.0) and no E12 lever:
    /// the descriptor advertises `additionalSpeakers`, never emotion or duration.
    func testCapabilitiesAreTTSWithATwoSpeakerCast() {
        XCTAssertEqual(Set(Dia2TTSPackage.manifest.capabilities), [.tts])
        let surface = Dia2TTSPackage.manifest.surfaces.first { $0.capability == .tts }
        XCTAssertEqual(surface?.ttsControls?.speakerTags, ["[S1]", "[S2]"])
        XCTAssertEqual(surface?.ttsControls?.maxSpeakers, 2)
        XCTAssertEqual(surface?.ttsControls?.emotionModes, [])
        XCTAssertEqual(surface?.ttsControls?.supportsTargetDuration, false)
        let names = surface?.parameters.map(\.name) ?? []
        XCTAssertTrue(names.contains("additionalSpeakers"))
        XCTAssertFalse(names.contains("emotion"))
        XCTAssertFalse(names.contains("targetDuration"))
    }

    func testProvenancePinsUpstream() {
        XCTAssertEqual(Dia2TTSPackage.manifest.provenance.sourceRepo, "nari-labs/Dia2-2B")
    }
}

final class MaterializationConformanceTests: XCTestCase {

    /// MAT-1..5: a fresh (dir-less) configuration reports its one source missing; an explicit path satisfies — for
    /// the published bf16 tier, and for the fp32 parity tier, which only ever loads from an explicit directory.
    func testMaterializationGatePerTier() throws {
        for quant in [Quant.bf16, .fp32] {
            let satisfied = try satisfiedConfiguration(quant: quant)
            defer { try? FileManager.default.removeItem(at: satisfied.modelDirectory!) }
            let report = MaterializationConformance.check(
                freshConfiguration: Dia2TTSConfiguration(quant: quant, modelsRootDirectory: emptyStoreRoot()),
                satisfiedConfiguration: satisfied)
            XCTAssertTrue(report.passed, "\(quant): \(report.summary)")
        }
    }

    /// The shipping tier materializes the mlx-community repo named by fleet convention (<upstream name>-<tier>).
    func testPublishedRepo() {
        XCTAssertEqual(Dia2TTSConfiguration().weightSources.map(\.repo), ["mlx-community/Dia2-2B-bf16"])
        XCTAssertEqual(Dia2TTSConfiguration().quant, .bf16)
    }

    /// The declared file list covers everything `Dia2Model.load` opens (config, both safetensors, the tokenizer).
    func testDeclaredFilesCoverEverythingLoadOpens() {
        let files = Set(Dia2TTSConfiguration.files)
        for f in ["config.json", "model.safetensors", "mimi.safetensors", "tokenizer.json", "tokenizer_config.json"] {
            XCTAssertTrue(files.contains(f), f)
        }
    }

    func testCodableExcludesEnvironmentURLs() throws {
        let c = Dia2TTSConfiguration(modelDirectory: URL(fileURLWithPath: "/tmp/x"), modelsRootDirectory: URL(fileURLWithPath: "/tmp/y"))
        let decoded = try JSONDecoder().decode(Dia2TTSConfiguration.self, from: JSONEncoder().encode(c))
        XCTAssertNil(decoded.modelDirectory)
        XCTAssertNil(decoded.modelsRootDirectory)
        XCTAssertEqual(decoded.repo, c.repo)
    }

    private func emptyStoreRoot() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("dia2-empty-store-\(UUID().uuidString)")
    }

    private func satisfiedConfiguration(quant: Quant) throws -> Dia2TTSConfiguration {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("dia2-explicit-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data().write(to: dir.appending(path: Dia2TTSConfiguration.probeFile))
        return Dia2TTSConfiguration(quant: quant, modelDirectory: dir)
    }
}

final class CancellationConformanceTests: XCTestCase {

    /// CAN-1/CAN-2 — a pre-cancelled `run()` surfaces `CancellationError` unchanged, even unloaded.
    func testPreCancelledRunPropagatesCancellation() async {
        let package = Dia2TTSPackage(configuration: Dia2TTSConfiguration())
        let report = await CancellationConformance.checkRun(package: package, request: TTSRequest(text: "[S1] cancellation probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    /// CAN-3 — checkpoints at every warmup and generated frame, with RunProgress on the same seam.
    func testCheckpointCadence() {
        let report = CancellationConformance.checkCadence(
            manifest: Dia2TTSPackage.manifest,
            posture: .cadence([.init(phase: .generate, unit: .frame, reportsRunProgress: true)]))
        XCTAssertTrue(report.passed, report.summary)
    }
}

final class RequestPlaneTests: XCTestCase {

    func testUnloadedPackageRejectsLegibly() async {
        let package = Dia2TTSPackage(configuration: Dia2TTSConfiguration())
        do {
            _ = try await package.run(TTSRequest(text: "[S1] hi"))
            XCTFail("expected notLoaded")
        } catch let error as PackageError {
            guard case .notLoaded = error else { return XCTFail("expected notLoaded, got \(error)") }
        } catch {
            XCTFail("expected PackageError, got \(error)")
        }
    }

    private func clip() -> Audio { Audio(format: .wav, data: Data([1, 2, 3]), sampleRate: 24_000, channels: 1) }

    /// Speaker 2 (1.49.0): the cast entry wins; `.auto` = a fresh voice; a clip needs its transcript; presets are
    /// refused; the deprecated metaData keys are read only when no cast entry is given.
    func testSpeakerTwoFromTheCast() throws {
        let s1 = TTSSpeakerVoice(voice: VoiceSelector(.referenceAudio(clip())), referenceTranscript: "one")
        let s2 = TTSSpeakerVoice(voice: VoiceSelector(.referenceAudio(clip())), referenceTranscript: "two")
        XCTAssertEqual(try Dia2TTSPackage.speaker2(of: TTSRequest(text: "[S1] a [S2] b", speakers: [s1, s2])),
                       .prefix(clip(), "two"))
        XCTAssertEqual(try Dia2TTSPackage.speaker2(of: TTSRequest(text: "x", speakers: [s1, TTSSpeakerVoice()])), .none)
        XCTAssertEqual(try Dia2TTSPackage.speaker2(of: TTSRequest(text: "x", speakers: [s1])), .none)
        XCTAssertThrowsError(try Dia2TTSPackage.speaker2(of: TTSRequest(
            text: "x", speakers: [s1, TTSSpeakerVoice(voice: VoiceSelector(.referenceAudio(clip())))])))
        XCTAssertThrowsError(try Dia2TTSPackage.speaker2(of: TTSRequest(
            text: "x", speakers: [s1, TTSSpeakerVoice(voice: VoiceSelector(.named("bob")))])))
        XCTAssertThrowsError(try Dia2TTSPackage.speaker2(of: TTSRequest(text: "x", speakers: [s1, s2, s2])))
        // Deprecated path: read only when the cast carries no second voice — and the cast wins when both are sent.
        let legacy: MetaData = ["speaker2Audio": .string(Data([9]).base64EncodedString()), "speaker2Transcript": .string("old")]
        XCTAssertEqual(try Dia2TTSPackage.speaker2(of: TTSRequest(text: "x", voice: s1.voice, referenceTranscript: "one",
                                                                  metaData: legacy)),
                       .prefix(Audio(format: .wav, data: Data([9]), sampleRate: nil, channels: nil), "old"))
        XCTAssertEqual(try Dia2TTSPackage.speaker2(of: TTSRequest(text: "x", voice: s1.voice, referenceTranscript: "one",
                                                                  additionalSpeakers: [s2], metaData: legacy)),
                       .prefix(clip(), "two"))
    }

    /// The ENGINE refuses a third voice before admission (1.49.0 declaration gate) — no weights are touched.
    func testEngineRefusesAThirdVoiceBeforeAdmission() async throws {
        let engine = MLXServeEngine()
        let store = FileManager.default.temporaryDirectory.appendingPathComponent("dia2-empty-\(UUID().uuidString)")
        let id = try await engine.register(Dia2TTSPackage.registration,
                                           configuration: Dia2TTSConfiguration(modelsRootDirectory: store))
        let v = TTSSpeakerVoice()
        do {
            _ = try await engine.run(TTSRequest(text: "[S1] a [S2] b [S3] c", speakers: [v, v, v]), package: id)
            XCTFail("a three-voice request was admitted")
        } catch let error as PackageError {
            guard case .unsupportedRequestFeature(let why) = error else { return XCTFail("\(error)") }
            XCTAssertTrue(why.contains("additionalSpeakers"), why)
        }
    }

    func testWordsParseFromMetaData() throws {
        let words = try Dia2TTSPackage.parseWords(.array([
            .object(["text": .string("Hello"), "start": .double(0.1), "end": .double(0.4)]),
            .object(["text": .string("there."), "start": .int(1), "end": .double(1.3)]),
        ]), field: "referenceWords")
        XCTAssertEqual(words, [Dia2PrefixWord(text: "Hello", start: 0.1, end: 0.4), Dia2PrefixWord(text: "there.", start: 1, end: 1.3)])
        XCTAssertThrowsError(try Dia2TTSPackage.parseWords(.string("no"), field: "referenceWords"))
        XCTAssertThrowsError(try Dia2TTSPackage.parseWords(.array([.object(["text": .string("x")])]), field: "referenceWords"))
    }
}

final class RuntimeUnitTests: XCTestCase {
    let ids = Dia2TokenIds(card: 49280, newWord: 2, pad: 3, bos: 1, zero: 7, spk1: 49152, spk2: 49153, audioPad: 2049, audioBos: 2048)

    /// The state machine against a hand-traced upstream run: forced padding after a two-token word, the second stream
    /// carrying the word two entries ahead, end_step at the first new-word past the script.
    func testStateMachineTrace() {
        let machine = Dia2StateMachine(ids: ids, secondStreamAhead: 2, maxPadding: 6, initialPadding: 0)
        let state = machine.newState([
            Dia2Entry(tokens: [49152, 100], text: "Hi", padding: 2),
            Dia2Entry(tokens: [200], text: "you", padding: 1),
            Dia2Entry(tokens: [300], text: "there", padding: 1),
        ])
        // step 0: new word → "Hi" consumed; main gets its first token, second = new_word
        XCTAssertEqual(tuple(machine.process(step: 0, state: state, token: 1)), [49152, 2, 1])
        // step 1: pending tokens force pad; main pops 100; second gets the lookahead (entry 2 ahead = "there")
        XCTAssertEqual(tuple(machine.process(step: 1, state: state, token: 1)), [100, 300, 0])
        // step 2: forced padding (2 → 1 left after step 1's decrement) keeps pad
        XCTAssertEqual(tuple(machine.process(step: 2, state: state, token: 1)), [3, 3, 0])
        // step 3: free → new word "you"
        XCTAssertEqual(tuple(machine.process(step: 3, state: state, token: 1)), [200, 2, 1])
        _ = machine.process(step: 4, state: state, token: 0)
        XCTAssertEqual(tuple(machine.process(step: 5, state: state, token: 1)), [300, 2, 1])
        _ = machine.process(step: 6, state: state, token: 0)
        XCTAssertNil(state.endStep)
        _ = machine.process(step: 7, state: state, token: 1)
        XCTAssertEqual(state.endStep, 7)
        XCTAssertEqual(state.transcript.map(\.0), ["Hi", "you", "there"])
        XCTAssertEqual(state.transcript.map(\.1), [0, 3, 5])
    }

    func tuple(_ t: (Int, Int, Bool)) -> [Int] { [t.0, t.1, t.2 ? 1 : 0] }

    /// The estimate keeps every word, stays inside the voiced span, is monotone, and pauses after punctuation.
    func testWordTimingEstimate() {
        let sr = 24_000
        var audio = [Float](repeating: 0, count: sr / 2)                                   // 0.5 s silence
        audio += (0 ..< 2 * sr).map { Float(sin(Double($0) * 0.05)) * 0.3 }               // 2 s "speech"
        audio += [Float](repeating: 0, count: sr / 2)
        let words = Dia2WordTiming.estimate(transcript: "Hello there, my friend. Bye", audio: audio, sampleRate: sr)
        XCTAssertEqual(words.map(\.text), ["Hello", "there,", "my", "friend.", "Bye"])
        XCTAssertEqual(words.first!.start, 0.5, accuracy: 0.03)
        XCTAssertEqual(words.last!.end, 2.5, accuracy: 0.03)
        for (a, b) in zip(words, words.dropFirst()) { XCTAssertLessThanOrEqual(a.end, b.start) }
        XCTAssertGreaterThan(words[2].start - words[1].end, 0.05)                         // the comma pause
    }

    /// CFG filter on toy logits: survivors are guided's top-k, the CONDITIONAL logits are kept, scale 1 is a no-op.
    func testGuidanceKeepsConditionalLogitsOfGuidedTopK() {
        Device.withDefaultDevice(.cpu) {
            let cond: [Float] = [1, 5, 2, 0, 4], uncond: [Float] = [0, 6, 0, 0, 1]
            let logits = MLXArray(cond + uncond, [2, 5])
            // guided = uncond + 2 (cond − uncond) = [2, 4, 4, 0, 7] → the 3rd largest is 4 → entries 1, 2, 4 survive
            let out = Dia2Guidance.apply(logits, scale: 2, filterK: 3).asArray(Float.self)
            XCTAssertEqual(out[4], 4); XCTAssertEqual(out[1], 5); XCTAssertEqual(out[2], 2)
            XCTAssertEqual(out[0], -.infinity); XCTAssertEqual(out[3], -.infinity)
            XCTAssertEqual(Dia2Guidance.apply(logits, scale: 1, filterK: 3).asArray(Float.self), cond)
            let masked = Dia2Guidance.maskAudio(MLXArray([Float](repeating: 0, count: 2050), [1, 2050]), pad: 2049, bos: 2048).asArray(Float.self)
            XCTAssertEqual(masked[2048], -Float.greatestFiniteMagnitude); XCTAssertEqual(masked[0], 0)
        }
    }
}
