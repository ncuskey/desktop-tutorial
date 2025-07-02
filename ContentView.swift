import SwiftUI

struct ContentView: View {
    @StateObject private var generator = MapGenerator()

    var body: some View {
        HStack {
            MapCanvasView(mesh: generator.mesh, map: generator.map)
                .frame(minWidth: 300, minHeight: 300)
            ControlsView(spacing: $generator.spacing,
                         rainfall: $generator.rainfall,
                         onGenerate: generator.regenerate)
                .frame(width: 200)
        }
        .onAppear { generator.regenerate() }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
