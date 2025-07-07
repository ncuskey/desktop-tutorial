import XCTest
@testable import MapEngine

final class MapEngineTests: XCTestCase {
    func testMakeMeshProducesBuffers() async throws {
        let generator = MapGenerator()
        let result = try await generator.makeMesh()
        XCTAssertFalse(result.mesh.points.isEmpty)
        XCTAssertFalse(result.mesh.triangles.isEmpty)
    }
}
