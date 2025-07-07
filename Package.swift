// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MapEngine",
    products: [
        // Library product exposing the map generation engine
        .library(
            name: "MapEngine",
            targets: ["MapEngine"]),
    ],
    targets: [
        // Core map generation module
        .target(
            name: "MapEngine"),
        // Unit tests for the map engine
        .testTarget(
            name: "MapEngineTests",
            dependencies: ["MapEngine"]
        ),
    ]
)
