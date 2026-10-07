// Validate.swift — the live wrapper lanes: Dia2TTSPackage load → run → unload with measured memory (the split
// footprint's evidence), the request plane's refusals, and a live mid-run cancel. Run on a QUIET box with the MLX pool
// at the engine's shipping cap (2 GiB, AB-L-0030): overlapping jobs skew both wall time and phys_footprint.

import Dia2Core
import Foundation
import MLX
import MLXDia2TTS
import MLXToolKit

func decodeWav16(_ data: Data) -> [Float] {
    guard data.count > 44 else { return [] }
    return data.subdata(in: 44 ..< data.count).withUnsafeBytes { raw in
        raw.bindMemory(to: Int16.self).map { Float(Int16(littleEndian: $0)) / 32767 }
    }
}

func dBFS(_ s: [Float]) -> Double {
    guard !s.isEmpty else { return -.infinity }
    let rms = sqrt(s.reduce(0) { $0 + Double($1 * $1) } / Double(s.count))
    return 20 * log10(max(rms, 1e-12))
}

func cueAudio(_ name: String) throws -> Audio {
    Audio(format: .wav, data: try Data(contentsOf: evalDir.appendingPathComponent("cues/\(name).wav")), sampleRate: 24_000, channels: 1)
}

func cueTranscript(_ name: String) throws -> String {
    let refs = try JSONSerialization.jsonObject(with: Data(contentsOf: evalDir.appendingPathComponent("cues/refs.json"))) as! [String: Any]
    return (refs[name] as! [String: Any])["text"] as! String
}

func packageConfiguration(_ quant: Quant) -> Dia2TTSConfiguration {
    Dia2TTSConfiguration(quant: quant, modelDirectory: weightsDir)
}

@MainActor func gateValidate(quant: Quant) async throws {
    print("[validate] Dia2TTSPackage (\(quant), \(weightsDir.lastPathComponent)) load → run → unload")
    MLX.Memory.cacheLimit = 2 * 1_073_741_824
    let package = Dia2TTSPackage(configuration: packageConfiguration(quant))
    let phys0 = physFootprintMB()
    let t0 = Date()
    try await package.load()
    let loadSecs = Date().timeIntervalSince(t0)
    let resident = mb(MLX.Memory.activeMemory), phys1 = physFootprintMB()
    print(String(format: "  load %.2fs  MLX active %.0f MB  phys %.0f → %.0f MB", loadSecs, resident, phys0, phys1))

    let s2 = try cueAudio("ref_s2")
    let prefixed: MetaData = ["seed": .int(3), "speaker2Audio": .string(s2.data.base64EncodedString()),
                              "speaker2Transcript": .string(try cueTranscript("ref_s2"))]
    let scene = "[S1] We should have left an hour ago. [S2] And whose fault is that? You couldn't find your keys. "
        + "[S1] They were in your coat pocket! [S2] Fine. Let's just go. The train won't wait."
    let long = String(repeating: "[S1] Did you hear that? [S2] Hear what? It's the wind. [S1] No, it was a voice. Somebody's down there. "
        + "[S2] Then we're not alone. Stay behind me. ", count: 3)
    let runs: [(String, TTSRequest)] = [
        ("line, auto", TTSRequest(text: "[S1] We don't have much time. The bridge goes down at midnight.", metaData: ["seed": .int(0)])),
        ("scene, auto", TTSRequest(text: scene, metaData: ["seed": .int(1)])),
        ("scene, both prefixed (estimated words)", TTSRequest(text: scene, voice: VoiceSelector(.referenceAudio(try cueAudio("ref_s1"))),
                                                              referenceTranscript: try cueTranscript("ref_s1"), metaData: prefixed)),
        ("scene again (cached prefix plan)", TTSRequest(text: scene, voice: VoiceSelector(.referenceAudio(try cueAudio("ref_s1"))),
                                                        referenceTranscript: try cueTranscript("ref_s1"), metaData: prefixed)),
        ("long scene ×3, auto", TTSRequest(text: long, metaData: ["seed": .int(2)])),
        ("near-max scene ×10, both prefixed", TTSRequest(text: String(repeating: scene + " ", count: 10),
                                                         voice: VoiceSelector(.referenceAudio(try cueAudio("ref_s1"))),
                                                         referenceTranscript: try cueTranscript("ref_s1"), metaData: prefixed)),
    ]
    var worstPeak = 0.0, worstPhys = 0.0
    for (i, (label, request)) in runs.enumerated() {
        MLX.Memory.peakMemory = 0
        let t = Date()
        let r = try await package.run(request) as! TTSResponse
        let wall = Date().timeIntervalSince(t)
        let s = decodeWav16(r.audio.data), secs = Double(s.count) / 24000
        let peak = mb(MLX.Memory.peakMemory), phys = physFootprintMB()
        worstPeak = max(worstPeak, peak); worstPhys = max(worstPhys, phys)
        let url = evalDir.appendingPathComponent("_out/swift-validate/\(quant.rawValue)_\(i).wav")
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try r.audio.data.write(to: url)
        print(String(format: "  run %d %@: %6.2fs audio in %6.2fs (RTF %.2f) · %.1f dBFS · MLX peak %.0f MB (activation %.0f MB) · cache %.0f MB · phys %.0f MB",
                     i, label, secs, wall, wall / max(secs, 0.01), dBFS(s), peak, peak - resident, mb(MLX.Memory.cacheMemory), phys))
        if secs < 0.5 || dBFS(s) < -45 { failures.append("validate run \(i): short or silent audio") }
    }
    // the request plane's refusals, each legible and before any compute
    let refusals: [(String, TTSRequest)] = [
        ("voice.named", TTSRequest(text: "[S1] hi", voice: VoiceSelector(.named("alice")))),
        ("referenceAudio without transcript", TTSRequest(text: "[S1] hi", voice: VoiceSelector(.referenceAudio(try cueAudio("ref_s1"))))),
        ("speaker2Audio without speaker 1", TTSRequest(text: "[S1] hi", metaData: ["speaker2Audio": .string(s2.data.base64EncodedString())])),
        ("empty text", TTSRequest(text: "   ")),
    ]
    for (label, request) in refusals {
        do {
            _ = try await package.run(request)
            failures.append("\(label) was not refused")
        } catch let e as PackageError {
            print("  ✓ refused \(label): \(e)")
        }
    }
    await package.unload()
    print(String(format: "  unload: MLX active %.0f MB cache %.0f MB phys %.0f MB", mb(MLX.Memory.activeMemory), mb(MLX.Memory.cacheMemory), physFootprintMB()))
    print(String(format: "[VAL] pkg=dia2-2b quant=%@ load=%.2fs resident_active=%.0fMB peak_active=%.0fMB activation=%.0fMB phys_max=%.0fMB",
                 quant.rawValue, loadSecs, resident, worstPeak, worstPeak - resident, worstPhys))
}

@MainActor func gateCancel() async throws {
    print("[cancel] live mid-run cancel through the package")
    MLX.Memory.cacheLimit = 2 * 1_073_741_824
    let package = Dia2TTSPackage(configuration: packageConfiguration(.bf16))
    try await package.load()
    _ = try await package.run(TTSRequest(text: "[S1] Warm up.", metaData: ["seed": .int(0)]))
    // the same seeded request before and after the cancels: identical samples = a cancel leaves no state behind
    let probe = TTSRequest(text: "[S1] Recovered after cancel. Everything is back to normal.", metaData: ["seed": .int(4)])
    let before = decodeWav16((try await package.run(probe) as! TTSResponse).audio.data)
    let s2 = try cueAudio("ref_s2")
    let request = TTSRequest(
        text: String(repeating: "[S1] Did you hear that? [S2] Hear what? It's the wind. [S1] No, it was a voice. ", count: 3),
        voice: VoiceSelector(.referenceAudio(try cueAudio("ref_s1"))), referenceTranscript: try cueTranscript("ref_s1"),
        metaData: ["seed": .int(1), "speaker2Audio": .string(s2.data.base64EncodedString()),
                   "speaker2Transcript": .string(try cueTranscript("ref_s2"))])
    for delay in [0.3, 2.5, 6.0] {   // lands in the prefix encode / warmup, early frames, late frames
        let t0 = Date()
        let task = Task { @InferenceActor in try await package.run(request) }
        try await Task.sleep(nanoseconds: UInt64(delay * 1e9))
        let cancelAt = Date().timeIntervalSince(t0)
        task.cancel()
        do {
            _ = try await task.value
            failures.append("cancel@\(delay)s: run completed despite cancel")
        } catch is CancellationError {
            let latency = (Date().timeIntervalSince(t0) - cancelAt) * 1000
            print(String(format: "  ✓ CancellationError (unwrapped) cancel@%.2fs latency %.0f ms", cancelAt, latency))
            if latency > 500 { failures.append(String(format: "cancel latency %.0f ms > 500 ms", latency)) }
        } catch {
            failures.append("cancel@\(delay)s: wrong error type \(error)")
        }
    }
    let after = decodeWav16((try await package.run(probe) as! TTSResponse).audio.data)
    let same = before.count == after.count && zip(before, after).allSatisfy { abs($0 - $1) <= 1.0 / 32767 }
    print(String(format: "  re-run after cancel: %.2fs audio, %.1f dBFS; before the cancels %.2fs, %.1f dBFS — %@",
                 Double(after.count) / 24000, dBFS(after), Double(before.count) / 24000, dBFS(before), same ? "identical" : "DIFFERENT"))
    if !same { failures.append("the post-cancel re-run differs from the same seeded request before the cancels") }
    let url = evalDir.appendingPathComponent("_out/swift-validate/cancel_rerun.wav")
    try writeWav16(after, sampleRate: 24_000, to: url)
    await package.unload()
}
