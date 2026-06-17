// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "SwiftMoE",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "SwiftMoE", targets: ["SwiftMoE"]),
        .executable(name: "swift-moe-server", targets: ["SwiftMoEServer"]),
        .executable(name: "swift-moe-chat", targets: ["SwiftMoEChat"]),
    ],
    dependencies: [
        .package(url: "https://github.com/swiftlang/swift-docc-plugin", from: "1.4.3"),
    ],
    targets: [
        // MARK: - C Interop Targets

        .target(
            name: "CTokenizer",
            path: "Sources/CTokenizer",
            publicHeadersPath: "include"
        ),
        .target(
            name: "CLineNoise",
            path: "Sources/CLineNoise",
            publicHeadersPath: "include"
        ),

        // MARK: - Main Library

        .target(
            name: "SwiftMoE",
            dependencies: [],
            path: "Sources/SwiftMoE"
        ),

        // MARK: - Executable Targets

        .executableTarget(
            name: "SwiftMoEServer",
            dependencies: ["SwiftMoE"],
            path: "Sources/SwiftMoEServer"
        ),
        .executableTarget(
            name: "SwiftMoEChat",
            dependencies: ["SwiftMoE", "CLineNoise"],
            path: "Sources/SwiftMoEChat"
        ),

        // MARK: - Tests

        .testTarget(
            name: "SwiftMoETests",
            dependencies: ["SwiftMoE"]
        ),
    ]
)
