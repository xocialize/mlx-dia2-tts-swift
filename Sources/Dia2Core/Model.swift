// Model.swift — dia2/core/model.py's Dia2Model (transformer + depformer under the checkpoint's own key paths), plus the
// runtime pieces build_runtime assembles around it: tokenizer, token ids, state machine, delays, Mimi.
//
// A model directory (Tools/oracle-capture/convert.py) holds config.json, model.safetensors (upstream keys verbatim),
// mimi.safetensors (kyutai/mimi on moshi-swift paths) and the tokenizer files.

import Dia2Mimi
import Foundation
import MLX
import MLXNN

public final class Dia2Network: Module {
    @ModuleInfo(key: "transformer") public var transformer: Dia2TransformerDecoder
    @ModuleInfo(key: "depformer") public var depformer: Dia2Depformer

    init(_ cfg: Dia2Config) {
        _transformer.wrappedValue = Dia2TransformerDecoder(cfg)
        _depformer.wrappedValue = Dia2Depformer(cfg)
    }
}

public final class Dia2Model: @unchecked Sendable {
    public let config: Dia2Config
    public let network: Dia2Network
    public let mimi: Mimi
    public let tokenizer: Dia2Tokenizer
    public let ids: Dia2TokenIds
    /// Activation dtype: float32 for the fp32 tier (the parity configuration), bfloat16 for the bf16 tier — upstream's
    /// CUDA "bfloat16" precision (norms and logits stay float32 either way).
    public let computeDType: DType
    public let sampleRate = 24_000
    public let frameRate = 12.5
    public var delays: [Int] { config.data.delayPattern }

    public init(config: Dia2Config, network: Dia2Network, mimi: Mimi, tokenizer: Dia2Tokenizer, computeDType: DType) {
        self.config = config
        self.network = network
        self.mimi = mimi
        self.tokenizer = tokenizer
        self.ids = tokenizer.tokenIds(config)
        self.computeDType = computeDType
    }

    /// `dtype` nil keeps the on-disk dtype (the tier decides) and computes in it.
    public static func load(directory: URL, dtype: DType? = nil) async throws -> Dia2Model {
        let config = try Dia2Config.load(directory: directory)
        guard config.audioChannels == MimiLoader.numCodebooks else {
            throw Dia2Error.badConfig("\(config.audioChannels) audio channels; the port expects Mimi's \(MimiLoader.numCodebooks)")
        }
        guard config.data.delayPattern.count == config.audioChannels else {
            throw Dia2Error.badConfig("delay_pattern has \(config.data.delayPattern.count) entries for \(config.audioChannels) channels")
        }
        let network = Dia2Network(config)
        let arrays = try WeightIO.load(directory.appendingPathComponent("model.safetensors"), dtype: dtype)
        let diskDType = arrays["transformer.layers.0.attn.q_proj.weight"]?.dtype ?? .float32
        var cast = arrays
        // RMSNorm weights stay float32 whatever the tier (upstream builds every norm `dtype=torch.float32`)
        for (k, v) in arrays where k.hasSuffix("norm.weight") && v.dtype != .float32 { cast[k] = v.asType(.float32) }
        try WeightIO.apply(cast, to: network, component: "model")
        let mimi = try MimiLoader.load(directory.appendingPathComponent("mimi.safetensors"))
        let tokenizer = try await Dia2Tokenizer.load(directory: directory)
        return Dia2Model(config: config, network: network, mimi: mimi, tokenizer: tokenizer, computeDType: dtype ?? diskDType)
    }

    public func stateMachine(initialPadding: Int) -> Dia2StateMachine {
        Dia2StateMachine(ids: ids, secondStreamAhead: config.data.secondStreamAhead, maxPadding: 6, initialPadding: initialPadding)
    }

    public func parse(_ script: String) -> [Dia2Entry] {
        Dia2ScriptParser.parse([Dia2Script.normalize(script)], tokenizer: tokenizer, ids: ids, frameRate: frameRate)
    }
}

public enum Dia2Script {
    /// `normalize_script` for a string: strip surrounding whitespace.
    public static func normalize(_ script: String) -> String { script.trimmingCharacters(in: .whitespacesAndNewlines) }
}
