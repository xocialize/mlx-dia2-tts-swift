// Tokenizer.swift — the script tokenizer. Upstream loads the checkpoint's GPT-2 BPE (SmolLM vocabulary, 49 204 ids
// plus the 52 added tokens: [S1], [S2] and the (laughs)-style cues) with `use_fast=False`; here swift-transformers
// reads the same files' tokenizer.json. The g1 gate holds the two to identical ids on the golden strings and on
// parse_script's entries.

import Foundation
import Tokenizers

public final class Dia2Tokenizer: Dia2TextTokenizer, @unchecked Sendable {
    let inner: Tokenizer

    public init(_ inner: Tokenizer) { self.inner = inner }

    public static func load(directory: URL) async throws -> Dia2Tokenizer {
        Dia2Tokenizer(try await AutoTokenizer.from(modelFolder: directory, strict: false))
    }

    public func encode(_ text: String) -> [Int] { inner.encode(text: text, addSpecialTokens: false) }

    public func id(_ token: String) -> Int? { inner.convertTokenToId(token) }

    /// `TokenIds` as build_runtime assembles them: bos is the tokenizer's bos id unless that is 0 / missing
    /// (`getattr(tokenizer, "bos_token_id", 1) or 1` — Dia2's bos is <|endoftext|> = 0, so 1); speakers fall back to
    /// new-word when the vocabulary lacks them.
    public func tokenIds(_ cfg: Dia2Config) -> Dia2TokenIds {
        let d = cfg.data
        let bos = inner.bosTokenId.flatMap { $0 == 0 ? nil : $0 } ?? 1
        return Dia2TokenIds(
            card: d.textVocabSize, newWord: d.textNewWordTokenId, pad: d.textPadTokenId, bos: bos, zero: d.textZeroTokenId,
            spk1: id("[S1]") ?? d.textNewWordTokenId, spk2: id("[S2]") ?? d.textNewWordTokenId,
            audioPad: d.audioPadTokenId, audioBos: d.audioBosTokenId)
    }
}
