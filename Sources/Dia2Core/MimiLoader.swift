// MimiLoader.swift — builds the lifted Kyutai Mimi (Dia2Mimi) for Dia2's 32 codebooks and loads the converted
// `mimi.safetensors` (Tools/oracle-capture/convert.py: kyutai/mimi re-keyed onto moshi-swift's module paths) under the
// refuse-partial-loads key contract.

import Dia2Mimi
import Foundation
import MLX
import MLXNN

public enum Dia2Error: Error, CustomStringConvertible {
    case missingFile(String)
    case badConfig(String)
    case keyContract(component: String, missing: [String], unused: [String])
    case invalidInput(String)

    public var description: String {
        switch self {
        case .missingFile(let p): return "Dia2: missing file \(p)"
        case .badConfig(let why): return "Dia2: bad config — \(why)"
        case .keyContract(let c, let m, let u):
            return "Dia2 \(c) key contract violated — missing \(m.count) \(m.prefix(5)), unused \(u.count) \(u.prefix(5))"
        case .invalidInput(let why): return "Dia2: invalid input — \(why)"
        }
    }
}

enum WeightIO {
    /// Loads a safetensors file on the CPU stream (the Metal-watchdog rule), casting float tensors to `dtype`.
    static func load(_ url: URL, dtype: DType?) throws -> [String: MLXArray] {
        guard FileManager.default.fileExists(atPath: url.path) else { throw Dia2Error.missingFile(url.path) }
        return try Device.withDefaultDevice(.cpu) { () throws -> [String: MLXArray] in
            var out = [String: MLXArray]()
            for (k, v) in try loadArrays(url: url) {
                let floating = [DType.float16, .bfloat16, .float32].contains(v.dtype)
                out[k] = (dtype != nil && floating && v.dtype != dtype!) ? v.asType(dtype!) : v
            }
            eval(Array(out.values))
            return out
        }
    }

    /// `derived`: reflected arrays computed at load (never on disk); `buffers`: on-disk tensors the module consumes
    /// through `update` without listing them as parameters. Everything else must match exactly.
    static func apply(_ arrays: [String: MLXArray], to module: Module, component: String,
                      derived: (String) -> Bool = { _ in false }, buffers: (String) -> Bool = { _ in false }) throws {
        let expected = Set(module.parameters().flattened().map(\.0).filter { !derived($0) })
        let onDisk = Set(arrays.keys)
        let missing = expected.subtracting(onDisk).sorted()
        let unused = onDisk.subtracting(expected).filter { !buffers($0) }.sorted()
        guard missing.isEmpty && unused.isEmpty else {
            throw Dia2Error.keyContract(component: component, missing: missing, unused: unused)
        }
        // noUnusedKeys: every on-disk tensor (the buffers included) must be consumed by some module
        try module.update(parameters: ModuleParameters.unflattened(arrays), verify: [.noUnusedKeys, .shapeMismatch])
        module.train(false)
        eval(module.parameters())
    }
}

public enum MimiLoader {
    public static let numCodebooks = 32

    public static func load(_ url: URL) throws -> Mimi {
        let mimi = Mimi(MimiConfig.mimi_2024_07(numCodebooks: numCodebooks), bSize: 1)
        // the lifted moshi-swift Mimi: ConvTransposed1d's `expandedWeight` is derived from `weight` at update; the
        // EMA codebooks load `embedding_sum` / `cluster_usage` / `_initialized` and derive the embedding.
        try WeightIO.apply(try WeightIO.load(url, dtype: .float32), to: mimi, component: "mimi",
                           derived: { $0.hasSuffix(".expandedWeight") },
                           buffers: { $0.contains("._codebook.") })
        return mimi
    }
}
