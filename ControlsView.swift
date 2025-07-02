import SwiftUI

struct ControlsView: View {
    @Binding var spacing: Float
    @Binding var rainfall: Float
    var onGenerate: () -> Void

    var body: some View {
        VStack {
            VStack(alignment: .leading) {
                Text("Spacing: \(spacing, specifier: "%.1f")")
                Slider(value: $spacing, in: 0...10, step: 0.1)
            }
            VStack(alignment: .leading) {
                Text("Rainfall: \(rainfall, specifier: "%.2f")")
                Slider(value: $rainfall, in: 0...1, step: 0.01)
            }
            Button("Generate", action: onGenerate)
        }
        .padding()
    }
}
