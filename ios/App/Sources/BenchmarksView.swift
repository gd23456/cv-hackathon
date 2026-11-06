import Charts
import KshanaCore
import SwiftUI

/// One measured configuration from the repo's evaluation results.
struct MeasuredRow: Decodable, Identifiable {
    let name: String
    let device: String
    let latencyMs: Double?
    let sizeMB: Double?
    let peakMB: Double?
    let clip: Double?
    let fid: Double?
    let note: String?
    var id: String { "\(name) \(device)" }
}

struct BenchmarksView: View {
    let studio: Studio
    @State private var rows: [MeasuredRow] = []

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Button {
                        Task { await studio.runBenchmark() }
                    } label: {
                        HStack {
                            Text(studio.isBenchmarking ? "Measuring 20 images" : "Measure this iPhone")
                            if studio.isBenchmarking {
                                Spacer()
                                ProgressView()
                            }
                        }
                    }
                    .disabled(studio.phase != .ready || studio.isBenchmarking)

                    if let result = studio.benchmark {
                        row("Per image, mean", String(format: "%.1f ms", result.meanMs))
                        row("Median", String(format: "%.1f ms", result.p50Ms))
                        row("95th percentile", String(format: "%.1f ms", result.p95Ms))
                        row("Peak memory", String(format: "%.0f MB", result.peakFootprintMB))
                        row("iOS", result.iOSVersion)
                        ShareLink(item: result.rowsCSV, preview: SharePreview("Benchmark CSV")) {
                            Label("Share results as CSV", systemImage: "square.and.arrow.up")
                        }
                    }
                } header: {
                    Text("Live on this phone")
                }

                let charted = rows.filter { $0.latencyMs != nil }
                if !charted.isEmpty {
                    Section {
                        Chart(charted) { row in
                            BarMark(
                                x: .value("Time per image (ms)", row.latencyMs ?? 0),
                                y: .value("Configuration", row.name)
                            )
                            .foregroundStyle(row.name.hasPrefix("Kshana") ? Color.kshanaAccent : Color.secondary.opacity(0.5))
                        }
                        .chartXScale(type: .log)
                        .chartXAxisLabel("Time per image (ms, log scale)")
                        .frame(height: CGFloat(charted.count) * 44 + 48)
                    } header: {
                        Text("Speed")
                    }
                }

                if !rows.isEmpty {
                    Section {
                        ForEach(rows) { row in
                            VStack(alignment: .leading, spacing: 3) {
                                Text(row.name).font(.subheadline.weight(.medium))
                                Text(row.device).font(.caption).foregroundStyle(.secondary)
                                Text(details(row)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                                if let note = row.note {
                                    Text(note).font(.caption2).foregroundStyle(.secondary)
                                }
                            }
                        }
                    } header: {
                        Text("Measured results")
                    } footer: {
                        Text("Numbers come from the project's evaluation runs. Results from different devices or methods aren't directly comparable.")
                    }
                }
            }
            .navigationTitle("Benchmarks")
            .task { rows = Self.loadRows() }
        }
    }

    private func row(_ title: String, _ value: String) -> some View {
        LabeledContent(title) {
            Text(value).monospacedDigit()
        }
    }

    private func details(_ row: MeasuredRow) -> String {
        var parts: [String] = []
        if let value = row.latencyMs { parts.append(String(format: "%.0f ms", value)) }
        if let value = row.sizeMB { parts.append(String(format: "%.0f MB", value)) }
        if let value = row.peakMB { parts.append(String(format: "peak %.0f MB", value)) }
        if let value = row.clip { parts.append(String(format: "CLIP %.2f", value)) }
        if let value = row.fid { parts.append(String(format: "FID %.1f", value)) }
        return parts.joined(separator: ", ")
    }

    private static func loadRows() -> [MeasuredRow] {
        guard
            let url = Bundle.main.url(forResource: "measured", withExtension: "json"),
            let data = try? Data(contentsOf: url)
        else { return [] }
        return (try? JSONDecoder().decode([MeasuredRow].self, from: data)) ?? []
    }
}
