import SwiftUI

struct CreateView: View {
    @Bindable var studio: Studio

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                imageArea
                TextField("Describe an image", text: $studio.prompt, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                Button("Generate") {
                    Task { await studio.generate() }
                }
                .buttonStyle(.borderedProminent)
                .disabled(studio.phase != .ready || studio.prompt.isEmpty)
                stats
            }
            .padding()
        }
    }

    @ViewBuilder private var imageArea: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 12).fill(.quaternary)
            if let image = studio.image {
                Image(uiImage: image).resizable().scaledToFit().clipShape(.rect(cornerRadius: 12))
            }
            switch studio.phase {
            case .preparing(let stage):
                ProgressView("Preparing: \(stage)")
            case .generating:
                ProgressView()
            case .failed(let message):
                Text(message).font(.footnote).foregroundStyle(.red).padding()
            case .ready:
                EmptyView()
            }
        }
        .aspectRatio(1, contentMode: .fit)
    }

    private var stats: some View {
        VStack(alignment: .leading, spacing: 4) {
            if let t = studio.timings {
                Text(verbatim: String(format: "Text %.0f ms, image model %.0f ms, decoder %.0f ms: %.0f ms", t.textMs, t.unetMs, t.decodeMs, t.totalMs))
            }
            if let load = studio.loadReport {
                Text(verbatim: String(format: "Model load %.1f s", load.totalMs / 1000))
            }
            Text(verbatim: String(format: "Memory %.0f MB, peak %.0f MB, available %.0f MB", studio.footprintMB, studio.peakMB, studio.availableMB))
        }
        .font(.caption.monospacedDigit())
        .foregroundStyle(.secondary)
    }
}
