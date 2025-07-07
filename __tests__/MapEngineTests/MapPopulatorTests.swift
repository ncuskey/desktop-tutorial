import XCTest
@testable import MapEngine

final class MapPopulatorTests: XCTestCase {
    func testPopulatorCreatesStatesAndRoads() async throws {
        let filePath = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("Tests/Fixtures/sample.map.json").path
        let map = try await MapPopulator.loadMapJSON(at: filePath)
        let states = MapPopulator.generateStates(from: map.cells, options: .init(count: 2))
        XCTAssertFalse(states.isEmpty)
        let roads = MapPopulator.generateRoads(from: map.towns, options: .init(maxDistance: 1.5))
        XCTAssertFalse(roads.isEmpty)
    }
}
