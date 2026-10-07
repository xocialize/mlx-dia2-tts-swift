// StateMachine.swift — dia2/runtime/state_machine.py and dia2/runtime/script_parser.py, ported 1:1.
//
// The decoder's action head says, each 80 ms frame, "new word" or "pad". The state machine turns that into the two text
// streams the next frame reads: when a new word is taken it queues that word's tokens (fed one per frame) and, with
// second_stream_ahead, the tokens of the word `ahead` entries later on the second stream; padding budgets force or forbid
// new words. When the script runs out, the frame of the next new-word becomes `end_step` (EOS).

import Foundation

public struct Dia2TokenIds: Sendable, Equatable {
    public var card: Int
    public var newWord: Int
    public var pad: Int
    public var bos: Int
    public var zero: Int
    public var spk1: Int
    public var spk2: Int
    public var audioPad: Int
    public var audioBos: Int
    public var ungenerated: Int = -2
}

public struct Dia2Entry: Sendable, Equatable {
    public var tokens: [Int]
    public var text: String
    public var padding: Int
}

public final class Dia2State {
    var entries: [Dia2Entry]           // front = next
    var paddingBudget: Int
    var forcedPadding: Int
    var pendingTokens: [Int] = []
    var lookaheadTokens: [Int] = []
    public internal(set) var endStep: Int? = nil
    var consumptionTimes: [Int] = []
    public internal(set) var transcript: [(String, Int)] = []

    init(entries: [Dia2Entry], initialPadding: Int) {
        self.entries = entries
        paddingBudget = initialPadding
        forcedPadding = initialPadding
    }

    /// `peek_tokens(count)`: the tokens of the count-th upcoming entry that has tokens.
    func peekTokens(_ count: Int) -> [Int] {
        var c = count
        for e in entries where !e.tokens.isEmpty {
            c -= 1
            if c == 0 { return e.tokens }
        }
        return []
    }
}

public struct Dia2StateMachine {
    public let ids: Dia2TokenIds
    public let secondStreamAhead: Int
    public let maxPadding: Int
    public var initialPadding: Int

    public init(ids: Dia2TokenIds, secondStreamAhead: Int, maxPadding: Int = 6, initialPadding: Int = 0) {
        self.ids = ids
        self.secondStreamAhead = secondStreamAhead
        self.maxPadding = maxPadding
        self.initialPadding = initialPadding
    }

    func newState(_ entries: [Dia2Entry]) -> Dia2State { Dia2State(entries: entries, initialPadding: initialPadding) }

    /// `process` → (main, second, consumedNewWord). second == -1 means "none" (the caller substitutes pad).
    func process(step: Int, state: Dia2State, token: Int, isForced: Bool = false) -> (Int, Int, Bool) {
        var t = sanitize(token)
        t = enforce(state, t, isForced)
        let (t2, consumed) = handleNewWord(step, state, t)
        let out = selectOutput(state, t2)
        let (main, second) = multiplex(state, out)
        return (main, second, consumed)
    }

    func sanitize(_ token: Int) -> Int {
        var t = token
        if t == 1 { t = ids.newWord } else if t == 0 { t = ids.pad }
        if t != ids.newWord && t != ids.pad { return ids.pad }
        return t
    }

    func enforce(_ s: Dia2State, _ token: Int, _ forced: Bool) -> Int {
        if !s.pendingTokens.isEmpty { return ids.pad }
        if forced { return token }
        if s.forcedPadding > 0 { return ids.pad }
        if s.paddingBudget <= 0 && token != ids.newWord { return ids.newWord }
        return token
    }

    func handleNewWord(_ step: Int, _ s: Dia2State, _ token: Int) -> (Int, Bool) {
        if token != ids.newWord { return (token, false) }
        if !s.entries.isEmpty {
            let entry = s.entries.removeFirst()
            s.consumptionTimes.append(step)
            var out = token
            if !entry.tokens.isEmpty {
                s.transcript.append((entry.text, step))
                s.pendingTokens.append(contentsOf: entry.tokens)
                if secondStreamAhead > 0 { s.lookaheadTokens.append(contentsOf: s.peekTokens(secondStreamAhead)) }
                s.paddingBudget = maxPadding
            } else {
                out = ids.pad
            }
            s.forcedPadding = entry.padding
            return (out, true)
        }
        var out = ids.pad
        if secondStreamAhead > 0 && s.endStep == nil { out = ids.newWord }
        if s.endStep == nil { s.endStep = step }
        return (out, false)
    }

    func selectOutput(_ s: Dia2State, _ token: Int) -> Int {
        if token == ids.pad {
            if s.paddingBudget > 0 { s.paddingBudget -= 1 }
            if s.forcedPadding > 0 { s.forcedPadding -= 1 }
            if !s.pendingTokens.isEmpty { return s.pendingTokens.removeFirst() }
            return ids.pad
        }
        if token == ids.newWord { return ids.newWord }
        if token == ids.zero { return token }
        preconditionFailure("invalid token \(token)")
    }

    func multiplex(_ s: Dia2State, _ output: Int) -> (Int, Int) {
        if secondStreamAhead == 0 { return (output, output) }
        var out = output
        let second: Int
        if out == ids.newWord {
            second = ids.newWord
            out = s.pendingTokens.isEmpty ? ids.pad : s.pendingTokens.removeFirst()
        } else if !s.lookaheadTokens.isEmpty {
            second = s.lookaheadTokens.removeFirst()
        } else {
            second = ids.pad
        }
        return (out, second)
    }
}

// MARK: - parse_script

public protocol Dia2TextTokenizer {
    func encode(_ text: String) -> [Int]       // add_special_tokens=False
}

public enum Dia2ScriptParser {
    /// `parse_script([script], tokenizer, constants, frame_rate)` — one entry per word; `[S1]` / `[S2]` set the speaker
    /// token prefixed to the next word; `<break time="1.5s"/>` becomes a tokenless padding entry.
    public static func parse(_ lines: [String], tokenizer: Dia2TextTokenizer, ids: Dia2TokenIds, frameRate: Double) -> [Dia2Entry] {
        var entries = [Dia2Entry]()
        let speakers = [ids.spk1, ids.spk2]
        let paddingBetween = 1
        var lastSpeakerIdx: Int? = nil
        let eventRe = try! NSRegularExpression(pattern: #"(?:<break\s+time="([0-9]+(?:.[0-9]*)?)s"\s*/?>)|(?:\s+)"#)

        for (idx, line) in lines.enumerated() {
            let normalized = line.replacingOccurrences(of: "\u{2019}", with: "'").replacingOccurrences(of: ":", with: " ")
            var firstContent = true
            var pending: Int? = nil

            func addEntry(_ word: String) {
                var tokens: [Int]
                if let p = pending {
                    tokens = tokenizer.encode("\(p == ids.spk1 ? "[S1]" : "[S2]") \(word)")
                } else {
                    tokens = tokenizer.encode(word)
                }
                if firstContent {
                    let speakerIdx = idx % speakers.count
                    let speakerToken = speakers[speakerIdx]
                    if lastSpeakerIdx != speakerIdx {
                        if tokens.first != speakerToken { tokens.insert(speakerToken, at: 0) }
                        lastSpeakerIdx = speakerIdx
                    }
                    firstContent = false
                }
                entries.append(Dia2Entry(tokens: tokens, text: word, padding: max(0, paddingBetween + tokens.count - 1)))
            }

            var remaining = normalized
            while !remaining.isEmpty {
                let ns = remaining as NSString
                let match = eventRe.firstMatch(in: remaining, range: NSRange(location: 0, length: ns.length))
                let segment: String
                if let m = match {
                    segment = ns.substring(to: m.range.location)
                    remaining = ns.substring(from: m.range.location + m.range.length)
                } else {
                    segment = remaining
                    remaining = ""
                }
                for raw in segment.split(whereSeparator: { $0.isWhitespace }).map(String.init) where !raw.isEmpty {
                    if raw == "[S1]" || raw == "[S2]" {
                        pending = raw == "[S1]" ? ids.spk1 : ids.spk2
                        continue
                    }
                    addEntry(raw)
                    pending = nil
                }
                if let m = match, m.range(at: 1).location != NSNotFound {
                    let seconds = Double(ns.substring(with: m.range(at: 1))) ?? 0
                    let padding = Int((seconds * frameRate).rounded(.toNearestOrEven))
                    if padding > 0 { entries.append(Dia2Entry(tokens: [], text: "", padding: padding)) }
                }
            }
        }
        return entries
    }
}
