// swift-tools-version: 6.2
// legibility:description: A Swift inference engine for Mixture-of-Experts language models on Apple Silicon, based on the Flash-MoE paper.

import PackageDescription

let package = Package(
    name: "SwiftMoE",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "SwiftMoE", targets: ["SwiftMoE"]),
        .library(name: "SwiftMoEHTTP", targets: ["SwiftMoEHTTP"]),
        .executable(name: "swift-moe-server", targets: ["SwiftMoEServer"]),
        .executable(name: "swift-moe-chat", targets: ["SwiftMoEChat"]),
    ],
    dependencies: [
        .package(url: "https://github.com/swiftlang/swift-docc-plugin", from: "1.4.3"),
        .package(url: "https://github.com/jpurnell/SwiftHTTPService.git", .upToNextMinor(from: "0.5.0")),
        // Tests only: a real TLS handshake against the server, with certificates made when the tests run.
        .package(url: "git@github.com:jpurnell/SwiftTLSKit.git", .upToNextMinor(from: "0.2.1")),
        .package(url: "https://github.com/apple/swift-nio.git", from: "2.102.0"),
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

        // MARK: - HTTP service (the OpenAI-compatible server, on SwiftHTTPService)

        .target(
            name: "SwiftMoEHTTP",
            dependencies: [
                "SwiftMoE",
                .product(name: "SwiftHTTPService", package: "SwiftHTTPService"),
                .product(name: "SwiftHTTPServiceNIO", package: "SwiftHTTPService"),
            ],
            path: "Sources/SwiftMoEHTTP"
        ),

        // MARK: - Executable Targets

        .executableTarget(
            name: "SwiftMoEServer",
            dependencies: ["SwiftMoE", "SwiftMoEHTTP", .product(name: "SwiftHTTPService", package: "SwiftHTTPService")],
            path: "Sources/SwiftMoEServer"
        ),
        .executableTarget(
            name: "SwiftMoEChat",
            dependencies: ["SwiftMoE", "SwiftMoEHTTP", "CLineNoise"],
            path: "Sources/SwiftMoEChat"
        ),

        // MARK: - Tests

        .testTarget(
            name: "SwiftMoETests",
            dependencies: [
                "SwiftMoE",
                "SwiftMoEHTTP",
                .product(name: "SwiftHTTPService", package: "SwiftHTTPService"),
                .product(name: "SwiftHTTPServiceNIO", package: "SwiftHTTPService"),
                .product(name: "SwiftTLSKit", package: "SwiftTLSKit"),
                .product(name: "SwiftTLSKitTestSupport", package: "SwiftTLSKit"),
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
            ],
            exclude: ["Fixtures"]
        ),
    ]
)
