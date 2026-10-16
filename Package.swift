// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "AgentSDK-Swift",
    platforms: [
        .macOS(.v13),
        .iOS(.v16)
    ],
    products: [
        .library(name: "Agents", targets: ["Agents"]),
        .library(name: "AgentsTesting", targets: ["AgentsTesting"])
    ],
    targets: [
        .target(name: "Agents"),
        .target(name: "AgentsTesting", dependencies: ["Agents"]),
        .executableTarget(
            name: "HelloWorld",
            dependencies: ["Agents"],
            path: "Examples/HelloWorld"
        ),
        .testTarget(
            name: "AgentsTests",
            dependencies: ["Agents", "AgentsTesting"]
        )
    ]
)
