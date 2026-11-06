// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "KshanaCore",
    platforms: [.iOS(.v18), .macOS(.v15)],
    products: [.library(name: "KshanaCore", targets: ["KshanaCore"])],
    targets: [
        .target(name: "KshanaCore"),
        .executableTarget(name: "kshana-cli", dependencies: ["KshanaCore"]),
        .testTarget(name: "KshanaCoreTests", dependencies: ["KshanaCore"]),
    ]
)
