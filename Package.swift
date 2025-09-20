// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "AgentSDK-Swift",
    platforms: [
        .macOS(.v13),
        .iOS(.v16)
    ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "AgentSDK-Swift",
            targets: ["AgentSDK-Swift"]),
        .executable(
            name: "SimpleApp",
            targets: ["SimpleApp"])
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-openapi-runtime", from: "1.0.0")
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "AgentSDK-Swift",
            dependencies: [
                .product(name: "OpenAPIRuntime", package: "swift-openapi-runtime")
            ]),
        .executableTarget(
            name: "SimpleApp",
            dependencies: ["AgentSDK-Swift"],
            path: "Examples"
        ),
        .testTarget(
            name: "AgentSDK-SwiftTests",
            dependencies: ["AgentSDK-Swift"]
        )
    ]
)
