import KshanaCore
import SwiftUI

/// Shows each stage of the last run, with real timings and intermediate images.
struct InsideView: View {
    let studio: Studio

    var body: some View {
        NavigationStack {
            List {
                Section("This iPhone") {
                    row("Memory now", String(format: "%.0f MB", studio.device.footprintMB))
                    row("Peak memory", String(format: "%.0f MB", studio.device.peakMB))
                    row("Still available", String(format: "%.0f MB", studio.device.availableMB))
                    row("Temperature", studio.device.thermalName)
                    row("Low Power Mode", studio.device.lowPower ? "On" : "Off")
                    if let load = studio.loadReport {
                        row("Model load", String(format: "%.1f s", load.totalMs / 1000))
                    }
                    row("Prompt checks", studio.verifierAvailable ? "On" : "Off")
                    if let acceleration = studio.acceleration {
                        row("Runs on", acceleration.isAccelerated ? "Neural Engine" : "CPU (slower)")
                    }
                }

                if let summary = studio.summary {
                    Section("Last image") {
                        row("Text encoder", ms(summary.best.timings.textMs))
                        row("Image model, 1 step", ms(summary.best.timings.unetMs))
                        row("Decoder", ms(summary.best.timings.decodeMs))
                        row("Checks", ms(summary.best.verifyMs))
                    }

                    Section("Search") {
                        row("Images tried", "\(summary.candidatesTried)")
                        row("Total time", String(format: "%.2f s", summary.elapsedMs / 1000))
                        row("Best seed", "\(summary.best.seed)")
                        row("Score", String(format: "%.2f", summary.best.score))
                        if let alignment = summary.best.alignment {
                            row("Prompt alignment", String(format: "%.3f", alignment))
                        }
                    }

                    Section("Quality measures") {
                        row("Sharpness", String(format: "%.5f", summary.best.quality.sharpness))
                        row("Contrast", String(format: "%.3f", summary.best.quality.contrast))
                        row("Colorfulness", String(format: "%.0f", summary.best.quality.colorfulness))
                        row("8-pixel grid", String(format: "%.2f", summary.best.quality.gridScore))
                        row("Clipped pixels", String(format: "%.1f%%", summary.best.quality.clippedFraction * 100))
                    }

                    Section("Post-processing") {
                        StageRow(name: "Generated", detail: "Straight from the decoder", timing: nil, image: summary.best.image)
                        ForEach(summary.stages) { stage in
                            StageRow(
                                name: stage.name, detail: stage.reason,
                                timing: stage.applied ? ms(stage.milliseconds) : nil, image: stage.image
                            )
                        }
                    }
                } else {
                    Section {
                        Text("Generate an image to see how it was made.")
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .scrollContentBackground(.hidden)
            .background { AuroraBackground(animated: studio.effectsAllowed) }
            .navigationTitle("Inside")
            .onAppear { studio.refreshDevice() }
        }
    }

    private func row(_ title: String, _ value: String) -> some View {
        LabeledContent(title) {
            Text(value).monospacedDigit()
        }
    }

    private func ms(_ value: Double) -> String {
        String(format: "%.1f ms", value)
    }
}

private struct StageRow: View {
    let name: String
    let detail: String
    let timing: String?
    let image: RGBAImage

    var body: some View {
        HStack(spacing: 12) {
            if let cgImage = image.cgImage {
                Image(uiImage: UIImage(cgImage: cgImage))
                    .resizable()
                    .frame(width: 64, height: 64)
                    .clipShape(.rect(cornerRadius: 8))
            }
            VStack(alignment: .leading, spacing: 2) {
                Text(name).font(.subheadline.weight(.medium))
                Text(detail).font(.caption).foregroundStyle(.secondary)
                if let timing {
                    Text(timing).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
            }
        }
    }
}
