// swift-tools-version: 6.2
// mlx-dia2-tts-swift — Nari Labs Dia2 (Apache-2.0) ported to Swift-MLX for MLXEngine: the fleet's two-speaker SCENE
// renderer. A `[S1]` / `[S2]` script becomes one 24 kHz take in which the two speakers trade turns; each speaker can be
// conditioned on a voice prefix (a clip + its transcript).
//
// Dia2 is a delayed-streams TTS: a main decoder (2B: 28 × 2048, GQA 16/8) reads the script word by word through an
// action stream, a 4-layer depformer emits all 32 Mimi codebooks per 12.5 Hz frame, and Kyutai's Mimi codec decodes.
//
//   • Dia2Mimi  — Kyutai's Mimi (encoder for prefixes, decoder for output), lifted from kyutai-labs/moshi-swift (MIT);
//                 changes marked "dia2:".
//   • Dia2Core  — the port: transformer, depformer, the word/action state machine, CFG + sampling, prefix plans,
//                 the generation loop of upstream nari-labs/dia2 runtime/generator.py.
//   • MLXDia2TTS — the engine-facing `tts` package.
//   • dia2-gates — parity gates against goldens from the upstream PyTorch runtime, plus render / validate lanes.
import PackageDescription

let package = Package(
    name: "mlx-dia2-tts-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "Dia2Mimi", targets: ["Dia2Mimi"]),
        .library(name: "Dia2Core", targets: ["Dia2Core"]),
        .library(name: "MLXDia2TTS", targets: ["MLXDia2TTS"]),
        .executable(name: "dia2-gates", targets: ["dia2-gates"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.5"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.64.0"),
    ],
    targets: [
        .target(
            name: "Dia2Mimi",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Sources/Dia2Mimi",
            exclude: ["LICENSE-moshi-swift.txt"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .target(
            name: "Dia2Core",
            dependencies: [
                "Dia2Mimi",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/Dia2Core",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .target(
            name: "MLXDia2TTS",
            dependencies: [
                "Dia2Core",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/MLXDia2TTS",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "MLXDia2TTSTests",
            dependencies: [
                "Dia2Core",
                "MLXDia2TTS",
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ],
            path: "Tests/MLXDia2TTSTests"
        ),
        .executableTarget(
            name: "dia2-gates",
            dependencies: [
                "Dia2Mimi", "Dia2Core", "MLXDia2TTS",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Sources/dia2-gates",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
