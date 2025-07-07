import Foundation

/// Dual mesh representation used by the map generator.
public struct Mesh {
    /// Vertex positions packed as x, y pairs.
    public var points: [Float]
    /// Indices for each triangle in the mesh.
    public var triangles: [Int]
    /// Region identifiers corresponding to each vertex.
    public var regions: [Int]
    /// Flags marking boundary triangles.
    public var isBoundary: [Bool]
    /// Elevation per triangle.
    public var elevationTriangles: [Float]
    /// Elevation per region.
    public var elevationRegions: [Float]

    public init(points: [Float] = [],
                triangles: [Int] = [],
                regions: [Int] = [],
                isBoundary: [Bool] = []) {
        self.points = points
        self.triangles = triangles
        self.regions = regions
        self.isBoundary = isBoundary
        self.elevationTriangles = Array(repeating: 0, count: triangles.count / 3)
        self.elevationRegions = Array(repeating: 0, count: regions.count)
    }
}

/// Parameters controlling elevation generation.
public struct ElevationParams {
    public var seed: UInt64
    public var mountainJaggedness: Float

    public init(seed: UInt64 = 0, mountainJaggedness: Float = 1.0) {
        self.seed = seed
        self.mountainJaggedness = mountainJaggedness
    }
}

/// Global constraints for map generation.
public struct Constraints {
    public var seaLevel: Float

    public init(seaLevel: Float = 0.0) {
        self.seaLevel = seaLevel
    }
}

/// Parameters controlling river generation.
public struct RiverParams {
    public var rainfall: Float

    public init(rainfall: Float = 0.5) {
        self.rainfall = rainfall
    }
}

/// Engine for generating procedural maps based on mapgen4.
public final class MapGenerator {
    private let queue = DispatchQueue(label: "MapEngine.background", qos: .userInitiated)

    public init() {}

    /// Builds the base mesh and picks peak triangles.
    public func makeMesh() async throws -> (mesh: Mesh, peaks: [Int]) {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                // TODO: Replace with Delaunay/Voronoi mesh construction
                let pointCount = 100
                var points: [Float] = []
                points.reserveCapacity(pointCount * 2)
                for _ in 0..<pointCount {
                    points.append(Float.random(in: 0...1))
                    points.append(Float.random(in: 0...1))
                }
                var triangles: [Int] = []
                for i in stride(from: 0, to: pointCount - 2, by: 3) {
                    triangles.append(i)
                    triangles.append(i + 1)
                    triangles.append(i + 2)
                }
                let regions = Array(0..<pointCount)
                let mesh = Mesh(points: points, triangles: triangles, regions: regions,
                                 isBoundary: Array(repeating: false, count: triangles.count / 3))
                let peaks = regions.prefix(5).map { $0 }
                continuation.resume(returning: (mesh, Array(peaks)))
            }
        }
    }

    /// Calculates elevation data for the provided mesh.
    public func assignElevation(mesh: inout Mesh, peaks: [Int], params: ElevationParams, constraints: Constraints) {
        queue.sync {
            // TODO: Implement elevation algorithm using params and constraints
            for index in mesh.elevationTriangles.indices {
                mesh.elevationTriangles[index] = Float.random(in: -1...1)
            }
            for index in mesh.elevationRegions.indices {
                mesh.elevationRegions[index] = mesh.elevationTriangles.first ?? 0
            }
        }
    }

    /// Generates river paths using the rainfall model.
    public func assignRivers(mesh: inout Mesh, params: RiverParams) {
        queue.sync {
            // TODO: Implement river algorithm using params
            _ = params.rainfall // placeholder to silence unused warning
        }
    }
}
