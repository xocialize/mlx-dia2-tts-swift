// Config.swift — Dia2's config.json (dia2/config.py `load_config`): the data block (stream layout, token ids, delay
// pattern), the model block (decoder / depformer geometry, RoPE timescales, norm epsilon) and the runtime block
// (depformer weights schedule, context length). Defaults mirror load_config's `.get(..., default)` fallbacks.

import Foundation

public struct Dia2DataConfig: Sendable {
    public var channels: Int
    public var textVocabSize: Int
    public var audioVocabSize: Int
    public var actionVocabSize: Int
    public var textPadTokenId: Int
    public var textNewWordTokenId: Int
    public var textZeroTokenId: Int
    public var audioPadTokenId: Int
    public var audioBosTokenId: Int
    public var actionPadTokenId: Int
    public var actionNewWordTokenId: Int
    public var delayPattern: [Int]
    public var firstWordMinStart: Int
    public var maxPad: Int
    public var secondStreamAhead: Int
}

public struct Dia2StackConfig: Sendable {
    public var nLayer: Int
    public var nEmbd: Int
    public var nHidden: Int
    public var queryHeads: Int
    public var kvHeads: Int
    public var headDim: Int
    public var applyRope: Bool
    public var textEmbedding: Bool
    public var mlpActivations: [String]
    public var lowRankDim: Int?
}

public struct Dia2Config: Sendable {
    public var data: Dia2DataConfig
    public var decoder: Dia2StackConfig
    public var depformer: Dia2StackConfig
    public var linearActivations: [String]
    public var ropeMinTimescale: Double
    public var ropeMaxTimescale: Double
    public var normEps: Float
    public var weightsSchedule: [Int]
    public var maxContextSteps: Int

    public var audioChannels: Int { max(0, data.channels - 2) }
    public var depth: Int { max(audioChannels - 1, 0) }

    public static func load(directory: URL) throws -> Dia2Config {
        let url = directory.appendingPathComponent("config.json")
        guard let data = FileManager.default.contents(atPath: url.path),
              let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let d = root["data"] as? [String: Any], let m = root["model"] as? [String: Any],
              let rt = root["runtime"] as? [String: Any],
              let dec = m["decoder"] as? [String: Any], let dep = m["depformer"] as? [String: Any]
        else { throw Dia2Error.badConfig("config.json is missing data / model / runtime blocks") }
        func int(_ o: [String: Any], _ k: String, _ def: Int? = nil) throws -> Int {
            if let v = o[k] as? Int { return v }
            if let def { return def }
            throw Dia2Error.badConfig("missing \(k)")
        }
        let audioVocab = try int(d, "audio_vocab_size")
        let dataCfg = Dia2DataConfig(
            channels: try int(d, "channels"), textVocabSize: try int(d, "text_vocab_size"), audioVocabSize: audioVocab,
            actionVocabSize: try int(d, "action_vocab_size"), textPadTokenId: try int(d, "text_pad_token_id"),
            textNewWordTokenId: try int(d, "text_new_word_token_id"), textZeroTokenId: try int(d, "text_zero_token_id", 7),
            audioPadTokenId: try int(d, "audio_pad_token_id", audioVocab - 1), audioBosTokenId: try int(d, "audio_bos_token_id", audioVocab - 2),
            actionPadTokenId: try int(d, "action_pad_token_id"), actionNewWordTokenId: try int(d, "action_new_word_token_id"),
            delayPattern: (d["delay_pattern"] as? [Int]) ?? [], firstWordMinStart: try int(d, "first_word_min_start", 0),
            maxPad: try int(d, "max_pad", 0), secondStreamAhead: try int(d, "second_stream_ahead", 0))
        let linear = ((m["linear"] as? [String: Any])?["mlp_activations"] as? [String]) ?? ["silu", "linear"]
        func stack(_ o: [String: Any], isDep: Bool) throws -> Dia2StackConfig {
            Dia2StackConfig(
                nLayer: try int(o, "n_layer"), nEmbd: try int(o, "n_embd"), nHidden: try int(o, "n_hidden"),
                queryHeads: try int(o, "gqa_query_heads"), kvHeads: try int(o, "kv_heads"), headDim: try int(o, "gqa_head_dim"),
                applyRope: isDep ? ((o["apply_rope"] as? Bool) ?? true) : true,
                textEmbedding: isDep ? ((o["text_embedding"] as? Bool) ?? true) : true,
                mlpActivations: isDep ? ((o["mlp_activations"] as? [String]) ?? ["silu", "linear"]) : linear,
                lowRankDim: o["low_rank_dim"] as? Int)
        }
        let audioChannels = max(0, dataCfg.channels - 2)
        let schedule = (rt["weights_schedule"] as? [Int]) ?? Array(0 ..< max(audioChannels - 1, 0))
        return Dia2Config(
            data: dataCfg, decoder: try stack(dec, isDep: false), depformer: try stack(dep, isDep: true), linearActivations: linear,
            ropeMinTimescale: (m["rope_min_timescale"] as? Double) ?? Double((m["rope_min_timescale"] as? Int) ?? 1),
            ropeMaxTimescale: (m["rope_max_timescale"] as? Double) ?? Double((m["rope_max_timescale"] as? Int) ?? 10000),
            normEps: Float((m["normalization_layer_epsilon"] as? Double) ?? 1e-5),
            weightsSchedule: schedule, maxContextSteps: (rt["max_context_steps"] as? Int) ?? 1500)
    }
}
