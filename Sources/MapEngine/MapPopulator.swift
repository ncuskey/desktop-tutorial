import Foundation

/// Container for map parsing and population utilities.
public enum MapPopulator {}

/// Map cell as parsed from Fantasy-Map-Generator data.
public struct Cell: Codable, Identifiable {
    /// Cell identifier matching index in the original mesh.
    public var id: Int
    /// X coordinate in map space.
    public var x: Double
    /// Y coordinate in map space.
    public var y: Double
    // TODO: Add additional metadata fields used by FMG

    public init(id: Int, x: Double, y: Double) {
        self.id = id
        self.x = x
        self.y = y
    }
}

/// Town / burg representation used for road generation.
public struct Town: Codable, Identifiable {
    public var id: Int
    public var cellId: Int
    public var x: Double
    public var y: Double
    public var name: String

    public init(id: Int, cellId: Int, x: Double, y: Double, name: String) {
        self.id = id
        self.cellId = cellId
        self.x = x
        self.y = y
        self.name = name
    }
}

/// State grouping multiple cells under a single government.
public struct State: Codable, Identifiable {
    public var id: Int
    public var name: String
    public var color: String
    public var cellIndices: [Int]
    public var capitalIndex: Int?

    public init(id: Int, name: String, color: String, cellIndices: [Int] = [], capitalIndex: Int? = nil) {
        self.id = id
        self.name = name
        self.color = color
        self.cellIndices = cellIndices
        self.capitalIndex = capitalIndex
    }
}

/// Connection between two towns.
public struct Road: Codable, Identifiable {
    public var id: Int { fromIndex }
    public var fromIndex: Int
    public var toIndex: Int
    public var type: String

    public init(fromIndex: Int, toIndex: Int, type: String) {
        self.fromIndex = fromIndex
        self.toIndex = toIndex
        self.type = type
    }
}

/// Options controlling state generation.
public struct StateOptions {
    public var count: Int
    public init(count: Int) { self.count = count }
}

/// Options controlling road generation.
public struct RoadOptions {
    public var maxDistance: Double
    public init(maxDistance: Double) { self.maxDistance = maxDistance }
}

/// Map data parsed from a .map JSON file.
public struct MapData: Codable {
    public var cells: [Cell]
    public var towns: [Town]
    public var states: [State]?
    public var roads: [Road]?
}

public extension MapPopulator {
    /// Load a .map JSON file from Fantasy-Map-Generator and decode into `MapData`.
    /// - Parameter path: File path to the JSON resource.
    /// - Returns: Parsed `MapData` instance.
    static func loadMapJSON(at path: String) async throws -> MapData {
        let url = URL(fileURLWithPath: path)
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global().async {
                do {
                    let data = try Data(contentsOf: url)
                    let mapData = try JSONDecoder().decode(MapData.self, from: data)
                    continuation.resume(returning: mapData)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Create a list of states by clustering cells.
    /// - Parameters:
    ///   - cells: All map cells.
    ///   - options: Generation options.
    /// - Returns: Array of generated states.
    static func generateStates(from cells: [Cell], options: StateOptions) -> [State] {
        guard !cells.isEmpty else { return [] }
        let count = max(1, min(options.count, cells.count))
        var states: [State] = (0..<count).map {
            State(id: $0, name: "State \($0)", color: String(format: "#%06X", Int.random(in: 0...0xFFFFFF)))
        }
        for (index, cell) in cells.enumerated() {
            states[index % count].cellIndices.append(cell.id)
        }
        for i in 0..<states.count {
            states[i].capitalIndex = states[i].cellIndices.first
        }
        // TODO: Improve clustering algorithm using k-means or Voronoi regions
        return states
    }

    /// Build a naive road network connecting nearby towns.
    /// - Parameters:
    ///   - towns: Town locations to connect.
    ///   - options: Parameters that influence road density.
    /// - Returns: List of roads.
    static func generateRoads(from towns: [Town], options: RoadOptions) -> [Road] {
        guard towns.count > 1 else { return [] }
        var roads: [Road] = []
        for (i, a) in towns.enumerated() {
            for b in towns[(i+1)...] {
                let dx = a.x - b.x
                let dy = a.y - b.y
                let distance = sqrt(dx * dx + dy * dy)
                if distance <= options.maxDistance {
                    roads.append(Road(fromIndex: a.id, toIndex: b.id, type: "road"))
                }
            }
        }
        // TODO: Use Delaunay triangulation for more natural routes
        return roads
    }
}

