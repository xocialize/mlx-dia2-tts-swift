import Foundation
import MLXToolKit

/// Init-time configuration for `Dia2TTSPackage` (C9). Per-request script / voices / metaData ride the canonical
/// `TTSRequest`.
///
/// ONE weight source backs the model: the converted checkpoint (Tools/oracle-capture/convert.py over
/// `nari-labs/Dia2-2B` @ main, Apache-2.0, and `kyutai/mimi`, CC-BY-4.0) — `model.safetensors` (transformer +
/// depformer, upstream keys), `mimi.safetensors` (re-keyed onto the lifted moshi-swift Mimi), `config.json` and the
/// tokenizer files.
///
/// `quant` is `.bf16` as-shipped (activations bf16; norms, logits and Mimi float32 — upstream's CUDA precision);
/// `.fp32` is the parity-gate tier.
public struct Dia2TTSConfiguration: PackageConfiguration, ModelStorable, QuantConfigured {
    /// The converted checkpoint repo (bf16 tier; the fp32 tier is `repo` with `-fp32`).
    public var repo: String
    /// Pinned revision; nil = main.
    public var revision: String?
    /// Compute tier: `.bf16` as-shipped, `.fp32` for parity work.
    public var quant: Quant
    /// Explicit checkpoint directory (dev escape hatch — never touches the network).
    public var modelDirectory: URL?
    /// Engine-chosen models root (auto-materialization target). Environment-specific.
    public var modelsRootDirectory: URL?

    public init(
        repo: String = "mlx-community/Dia2-2B-bf16",
        revision: String? = nil,
        quant: Quant = .bf16,
        modelDirectory: URL? = nil,
        modelsRootDirectory: URL? = nil
    ) {
        self.repo = repo
        self.revision = revision
        self.quant = quant
        self.modelDirectory = modelDirectory
        self.modelsRootDirectory = modelsRootDirectory
    }

    /// The repo for the configured tier: the fp32 tier lives beside the bf16 one.
    public var tierRepo: String {
        quant == .fp32 ? repo.replacingOccurrences(of: "-bf16", with: "-fp32") : repo
    }

    // Environment-specific URLs are excluded from Codable.
    private enum CodingKeys: String, CodingKey { case repo, revision, quant }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        repo = try c.decode(String.self, forKey: .repo)
        revision = try c.decodeIfPresent(String.self, forKey: .revision)
        quant = try c.decode(Quant.self, forKey: .quant)
    }
}

// MARK: - Weight sources (auto-materialization, engine MAT gate)

extension Dia2TTSConfiguration: WeightSourcing {
    static let files = ["config.json", "model.safetensors", "mimi.safetensors", "tokenizer.json", "tokenizer_config.json",
                        "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json"]
    /// Representative file for the missing-probe (the largest; a partial download is most likely to lack it).
    static let probeFile = "model.safetensors"

    public var weightSources: [WeightSource] {
        [WeightSource(role: "main", repo: tierRepo, revision: revision, matching: Self.files)]
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        let fm = FileManager.default
        if let modelDirectory, fm.fileExists(atPath: modelDirectory.appending(path: Self.probeFile).path) { return [] }
        if let dir = ModelStore(root: storeRoot).directory(for: tierRepo),
           fm.fileExists(atPath: dir.appending(path: Self.probeFile).path) { return [] }
        return weightSources
    }

    /// The configuration with a nil directory resolved to the store layout — what `load()` uses AFTER
    /// materialization. An explicit directory always wins.
    public func resolved(storeRoot: URL?) -> Dia2TTSConfiguration {
        var cfg = self
        if cfg.modelDirectory == nil { cfg.modelDirectory = ModelStore(root: storeRoot).directory(for: tierRepo) }
        return cfg
    }
}

// MARK: - Cold-start prewarm

extension Dia2TTSConfiguration: WeightPrewarming {
    public var prewarmPaths: [URL] {
        guard let dir = resolved(storeRoot: modelsRootDirectory).modelDirectory else { return [] }
        return ["model.safetensors", "mimi.safetensors"].map { dir.appending(path: $0) }
    }
}
