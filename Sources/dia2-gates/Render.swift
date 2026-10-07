// Render.swift — GPU render lanes: the E23 job list through the port (same wav + manifest.jsonl layout as upstream's
// run_dia2_v0.py, so harness/score_dia.py scores both), and a one-off render. Prefix word timings come from
// cues/refs.json keyed by the clip's file stem, as the oracle does. Run on a QUIET box.

import AVFoundation
import Darwin
import Dia2Core
import Foundation
import MLX

func physFootprintMB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
    }
    return result == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : -1
}

func mb(_ bytes: Int) -> Double { Double(bytes) / 1_048_576 }

/// Mono float samples of a 24 kHz WAV (the cue clips are 24 kHz PCM16 — anything else is refused, not resampled).
func readWav24k(_ url: URL) throws -> [Float] {
    let file = try AVAudioFile(forReading: url)
    guard file.fileFormat.sampleRate == 24_000 else { throw Dia2Error.invalidInput("\(url.lastPathComponent): \(file.fileFormat.sampleRate) Hz, need 24 kHz") }
    let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 24_000, channels: file.fileFormat.channelCount, interleaved: false)!
    let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(file.length))!
    try file.read(into: buf)
    let n = Int(buf.frameLength), ch = Int(format.channelCount)
    return (0 ..< n).map { i in (0 ..< ch).reduce(Float(0)) { $0 + buf.floatChannelData![$1][i] } / Float(ch) }
}

/// upstream write_wav: clip to [-1, 1], ×32767, PCM16 mono.
func writeWav16(_ samples: [Float], sampleRate: Int, to url: URL) throws {
    var d = Data()
    func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
    func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
    d.append(contentsOf: Array("RIFF".utf8)); u32(UInt32(36 + samples.count * 2)); d.append(contentsOf: Array("WAVEfmt ".utf8))
    u32(16); u16(1); u16(1); u32(UInt32(sampleRate)); u32(UInt32(sampleRate * 2)); u16(2); u16(16)
    d.append(contentsOf: Array("data".utf8)); u32(UInt32(samples.count * 2))
    for s in samples { u16(UInt16(bitPattern: Int16(max(-1, min(1, s)) * 32767))) }
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try d.write(to: url)
}

final class PrefixCache {
    let model: Dia2Model
    let refs: [String: Any]
    /// Estimate the prefix word timings from the transcript (`Dia2WordTiming`) instead of refs.json's Whisper times —
    /// the package's path when a caller sends no timings.
    let estimateWords: Bool
    var plans = [String: Dia2PrefixPlan]()

    init(_ model: Dia2Model, estimateWords: Bool = false) throws {
        self.model = model
        self.estimateWords = estimateWords
        refs = try JSONSerialization.jsonObject(with: Data(contentsOf: evalDir.appendingPathComponent("cues/refs.json"))) as! [String: Any]
    }

    func voice(_ path: String) throws -> Dia2Voice {
        let stem = URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent
        guard let r = refs[stem] as? [String: Any], let ws = r["words"] as? [[String: Any]] else { throw Dia2Error.invalidInput("no words for \(stem) in refs.json") }
        let audio = try readWav24k(URL(fileURLWithPath: path))
        if estimateWords {
            let words = Dia2WordTiming.estimate(transcript: r["text"] as! String, audio: audio, sampleRate: 24_000)
            print("    \(stem): estimated \(words.count) word timings (refs.json has \(ws.count))")
            return Dia2Voice(audio: audio, words: words)
        }
        let words = ws.map { Dia2PrefixWord(text: $0["text"] as! String, start: ($0["start"] as! NSNumber).doubleValue, end: ($0["end"] as! NSNumber).doubleValue) }
        return Dia2Voice(audio: audio, words: words)
    }

    func plan(s1: String?, s2: String?) throws -> Dia2PrefixPlan? {
        guard let s1 else { return nil }
        let key = "\(s1)|\(s2 ?? "")"
        if let p = plans[key] { return p }
        let p = model.prefixPlan(s1: try voice(s1), s2: try s2.map(voice))
        plans[key] = p
        return p
    }
}

/// `--render-jobs FILE --system NAME`: one JSON job per line (run_dia2_v0.py's fields); resumes from the manifest.
func renderJobs(_ m: Dia2Model, jobsFile: URL, system: String, cacheLimitMB: Int, estimateWords: Bool = false) throws {
    // MLX's buffer pool defaults to the memory limit (≫ 100 GB here) and never trims on its own: a long render
    // grows phys_footprint by the pool, not by live arrays. Cap it as the package does.
    MLX.Memory.cacheLimit = cacheLimitMB * 1_048_576
    let out = evalDir.appendingPathComponent("_out/\(system)")
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    let manifest = out.appendingPathComponent("manifest.jsonl")
    let done = Set(((try? String(contentsOf: manifest, encoding: .utf8)) ?? "").split(separator: "\n").compactMap {
        (try? JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: Any])?["path"] as? String
    })
    let jobs = try String(contentsOf: jobsFile, encoding: .utf8).split(separator: "\n").map {
        try JSONSerialization.jsonObject(with: Data($0.utf8)) as! [String: Any]
    }
    let prefixes = try PrefixCache(m, estimateWords: estimateWords)
    let resident = mb(MLX.Memory.activeMemory)
    var worstPeak = 0.0, worstPhys = 0.0, totalWall = 0.0, totalAudio = 0.0
    print(String(format: "render %d jobs → %@ (resident %.0f MB, phys %.0f MB)", jobs.count, out.path, resident, physFootprintMB()))
    for (i, j) in jobs.enumerated() {
        let set = j["set"] as! String, id = j["id"] as! String, cond = j["cond"] as! String, seed = j["seed"] as! Int
        let path = out.appendingPathComponent("\(set)/\(id)_\(cond)_s\(seed).wav")
        if done.contains(path.path) { continue }
        MLX.Memory.peakMemory = 0
        let t0 = Date()
        let plan = try prefixes.plan(s1: j["prefix_s1"] as? String, s2: j["prefix_s2"] as? String)
        let r = try m.generate(j["script"] as! String, prefix: plan, sampler: Dia2RandomSampler(seed: UInt64(seed)))
        let wall = Date().timeIntervalSince(t0)
        try writeWav16(r.waveform, sampleRate: r.sampleRate, to: path)
        let dur = Double(r.waveform.count) / Double(r.sampleRate)
        let peak = mb(MLX.Memory.peakMemory), phys = physFootprintMB()
        worstPeak = max(worstPeak, peak); worstPhys = max(worstPhys, phys); totalWall += wall; totalAudio += dur
        var row = j.filter { $0.key != "turns" }
        row["system"] = system; row["lang"] = "en"; row["path"] = path.path
        row["wall_s"] = (wall * 100).rounded() / 100; row["dur_s"] = (dur * 1000).rounded() / 1000
        row["sample_rate"] = r.sampleRate; row["n_frames"] = r.audioTokens.dim(1)
        row["words"] = r.timestamps.map { [$0.word, ($0.seconds * 1000).rounded() / 1000] as [Any] }
        row["turns"] = j["turns"] ?? NSNull()
        let line = try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys, .withoutEscapingSlashes])
        let h = try FileHandle(forWritingTo: manifest.existsOrCreated())
        h.seekToEndOfFile(); h.write(line); h.write(Data("\n".utf8)); try h.close()
        print(String(format: "%d/%d %@ %.2f s in %.2f s (RTF %.2f) · %d frames · peak %.0f MB · cache %.0f MB · phys %.0f MB",
                     i + 1, jobs.count, path.lastPathComponent, dur, wall, wall / max(dur, 0.01), r.frames, peak,
                     mb(MLX.Memory.cacheMemory), phys))
        fflush(stdout)
    }
    print(String(format: "[RENDER] system=%@ audio=%.1fs wall=%.1fs RTF=%.2f resident=%.0fMB peak_active=%.0fMB activation=%.0fMB phys_max=%.0fMB",
                 system, totalAudio, totalWall, totalWall / max(totalAudio, 0.01), resident, worstPeak, worstPeak - resident, worstPhys))
}

extension URL {
    func existsOrCreated() -> URL {
        if !FileManager.default.fileExists(atPath: path) { FileManager.default.createFile(atPath: path, contents: nil) }
        return self
    }
}
