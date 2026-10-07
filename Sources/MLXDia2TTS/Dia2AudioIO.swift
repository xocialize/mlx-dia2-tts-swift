import AVFoundation
import CryptoKit
import Foundation
import MLXToolKit

/// Canonical `Audio` (.wav) in and out. Voice prefixes are decoded with AVFoundation (any valid WAV layout), mixed to
/// mono and converted to Mimi's 24 kHz with AVAudioConverter (upstream resamples with sphn; the codes of a resampled
/// clip are not bit-identical across resamplers, which only matters off the 24 kHz path).
enum Dia2AudioIO {
    static let sampleRate = 24_000

    static func decode24k(_ audio: Audio) throws -> [Float] {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("dia2-ref-\(UUID().uuidString).wav")
        try audio.data.write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let file: AVAudioFile
        do { file = try AVAudioFile(forReading: tmp) } catch {
            throw PackageError.unsupportedRequestFeature("voice prefix clip is not a readable .wav (\(error.localizedDescription))")
        }
        let src = file.processingFormat
        let frames = AVAudioFrameCount(file.length)
        guard frames > 0, let inBuf = AVAudioPCMBuffer(pcmFormat: src, frameCapacity: frames) else {
            throw PackageError.unsupportedRequestFeature("voice prefix clip is empty")
        }
        try file.read(into: inBuf)
        let mono = mixToMono(inBuf)
        if Int(src.sampleRate) == sampleRate { return mono }
        return try resample(mono, from: src.sampleRate)
    }

    static func mixToMono(_ buf: AVAudioPCMBuffer) -> [Float] {
        let ch = Int(buf.format.channelCount), n = Int(buf.frameLength)
        guard let data = buf.floatChannelData, ch > 0 else { return [] }
        var mono = [Float](repeating: 0, count: n)
        for c in 0 ..< ch { for i in 0 ..< n { mono[i] += data[c][i] } }
        if ch > 1 { let inv = 1 / Float(ch); for i in 0 ..< n { mono[i] *= inv } }
        return mono
    }

    static func resample(_ mono: [Float], from rate: Double) throws -> [Float] {
        let inFmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: rate, channels: 1, interleaved: false)!
        let outFmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1, interleaved: false)!
        guard let conv = AVAudioConverter(from: inFmt, to: outFmt),
              let inBuf = AVAudioPCMBuffer(pcmFormat: inFmt, frameCapacity: AVAudioFrameCount(mono.count)) else {
            throw PackageError.unsupportedRequestFeature("cannot resample the voice prefix clip from \(rate) Hz")
        }
        conv.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        inBuf.frameLength = AVAudioFrameCount(mono.count)
        mono.withUnsafeBufferPointer { inBuf.floatChannelData![0].update(from: $0.baseAddress!, count: mono.count) }
        let capacity = AVAudioFrameCount((Double(mono.count) * Double(sampleRate) / rate).rounded(.up)) + 1024
        guard let outBuf = AVAudioPCMBuffer(pcmFormat: outFmt, frameCapacity: capacity) else {
            throw PackageError.unsupportedRequestFeature("cannot allocate the resampled voice prefix")
        }
        var fed = false
        var err: NSError?
        let status = conv.convert(to: outBuf, error: &err) { _, outStatus in
            if fed { outStatus.pointee = .endOfStream; return nil }
            fed = true
            outStatus.pointee = .haveData
            return inBuf
        }
        guard status != .error else {
            throw PackageError.unsupportedRequestFeature("voice prefix resample failed (\(err?.localizedDescription ?? "?"))")
        }
        return Array(UnsafeBufferPointer(start: outBuf.floatChannelData![0], count: Int(outBuf.frameLength)))
    }

    /// Mono float [-1, 1] → 16-bit PCM WAV (upstream write_wav's scaling: × 32767).
    static func encodeWAV16(samples: [Float], sampleRate: Int) -> Data {
        var d = Data()
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        let bytes = samples.count * 2
        d.append(contentsOf: Array("RIFF".utf8)); u32(UInt32(36 + bytes)); d.append(contentsOf: Array("WAVE".utf8))
        d.append(contentsOf: Array("fmt ".utf8)); u32(16); u16(1); u16(1); u32(UInt32(sampleRate)); u32(UInt32(sampleRate * 2)); u16(2); u16(16)
        d.append(contentsOf: Array("data".utf8)); u32(UInt32(bytes))
        d.reserveCapacity(d.count + bytes)
        for s in samples {
            let v = Int16(max(-1, min(1, s.isFinite ? s : 0)) * 32767)
            withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) }
        }
        return d
    }

    static func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
