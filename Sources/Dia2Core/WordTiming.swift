// WordTiming.swift — word timings for a voice prefix when the caller has only its transcript.
//
// Upstream times a prefix's words with whisper_timestamped (AGPL — never installed here); the timings become the frames
// at which the warmup forces "new word" on the action stream. When a caller supplies no timings (the canonical
// `TTSRequest` carries the transcript, not word times), they are ESTIMATED: the clip's voiced span (frame energy within
// 35 dB of its loudest 20 ms) is shared among the words in proportion to their letters, with a pause after clause and
// sentence punctuation. The E23 port lane measures what the estimate costs against aligner timings (MEASUREMENTS.md).

import Foundation

public enum Dia2WordTiming {
    /// Pause weights, in letters, after a word ending in clause / sentence punctuation.
    static let clausePause = 3.0
    static let sentencePause = 6.0

    public static func estimate(transcript: String, audio: [Float], sampleRate: Int) -> [Dia2PrefixWord] {
        let words = transcript.split(whereSeparator: { $0.isWhitespace }).map(String.init)
        guard !words.isEmpty, !audio.isEmpty, sampleRate > 0 else { return [] }
        let (start, end) = voicedSpan(audio, sampleRate: sampleRate)
        let letters = words.map { w in Double(max(1, w.unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) }.count)) }
        let pauses = words.enumerated().map { i, w -> Double in
            guard i < words.count - 1, let last = w.last else { return 0 }
            if ".!?…".contains(last) { return sentencePause }
            if ",;:—–".contains(last) { return clausePause }
            return 0
        }
        let unit = (end - start) / max(letters.reduce(0, +) + pauses.reduce(0, +), 1)
        var cursor = start
        return zip(words, zip(letters, pauses)).map { w, lp in
            let s = cursor, e = cursor + lp.0 * unit
            cursor = e + lp.1 * unit
            return Dia2PrefixWord(text: w, start: (s * 100).rounded() / 100, end: (e * 100).rounded() / 100)
        }
    }

    /// Start and end (seconds) of the clip's voiced span: 20 ms frames whose RMS is within 35 dB of the loudest.
    static func voicedSpan(_ audio: [Float], sampleRate: Int) -> (Double, Double) {
        let hop = max(1, sampleRate / 50)
        let frames = stride(from: 0, to: audio.count, by: hop).map { i -> Float in
            let seg = audio[i ..< min(i + hop, audio.count)]
            return (seg.reduce(0) { $0 + $1 * $1 } / Float(seg.count)).squareRoot()
        }
        let threshold = max((frames.max() ?? 0) * powf(10, -35 / 20), 1e-4)
        guard let first = frames.firstIndex(where: { $0 >= threshold }), let last = frames.lastIndex(where: { $0 >= threshold }) else {
            return (0, Double(audio.count) / Double(sampleRate))
        }
        return (Double(first * hop) / Double(sampleRate), Double(min((last + 1) * hop, audio.count)) / Double(sampleRate))
    }
}
