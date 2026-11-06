import Foundation

/// Repeated end-to-end generation on this device, with warm-up excluded.
public struct BenchmarkResult: Sendable {
    public let runs: Int
    public let meanMs: Double
    public let p50Ms: Double
    public let p95Ms: Double
    public let unetMeanMs: Double
    public let decodeMeanMs: Double
    public let textMeanMs: Double
    public let peakFootprintMB: Double
    public let thermalStart: Int
    public let thermalEnd: Int
    public let iOSVersion: String
    public let rowsCSV: String

    public var summaryLine: String {
        String(
            format: "benchmark %d runs ms: mean %.1f, p50 %.1f, p95 %.1f | text %.1f, unet %.1f, decode %.1f | peak footprint %.0f MB | thermal %d to %d",
            runs, meanMs, p50Ms, p95Ms, textMeanMs, unetMeanMs, decodeMeanMs, peakFootprintMB, thermalStart, thermalEnd
        )
    }

    public static func measure(generator: Generator, runs: Int) async throws -> BenchmarkResult {
        let prompts = [
            "a red apple to the left of a blue cup on a wooden table",
            "a close-up picture of an old man standing in the rain",
            "a golden retriever puppy sitting on green grass, sunny day",
        ]
        let thermalStart = ProcessInfo.processInfo.thermalState.rawValue
        // One warm-up image: the first call pays for lazy setup.
        _ = try await generator.generate(prompt: prompts[0], seed: 1)

        var timings: [GenerationTimings] = []
        var csv = "run,prompt,text_ms,unet_ms,decode_ms,total_ms,footprint_mb,thermal\n"
        for run in 0..<runs {
            let candidate = try await generator.generate(prompt: prompts[run % prompts.count], seed: 1000 + UInt64(run))
            let t = candidate.timings
            timings.append(t)
            let footprint = Double(MemoryProbe.snapshot()?.footprintBytes ?? 0) / 1_048_576
            csv += String(format: "%d,%d,%.2f,%.2f,%.2f,%.2f,%.0f,%d\n", run, run % prompts.count, t.textMs, t.unetMs, t.decodeMs, t.totalMs, footprint, ProcessInfo.processInfo.thermalState.rawValue)
        }

        let totals = timings.map(\.totalMs).sorted()
        func mean(_ values: [Double]) -> Double { values.reduce(0, +) / Double(max(1, values.count)) }
        func percentile(_ p: Double) -> Double { totals[min(totals.count - 1, Int(Double(totals.count) * p))] }
        return BenchmarkResult(
            runs: runs, meanMs: mean(totals), p50Ms: percentile(0.5), p95Ms: percentile(0.95),
            unetMeanMs: mean(timings.map(\.unetMs)), decodeMeanMs: mean(timings.map(\.decodeMs)), textMeanMs: mean(timings.map(\.textMs)),
            peakFootprintMB: Double(MemoryProbe.snapshot()?.lifetimePeakBytes ?? 0) / 1_048_576,
            thermalStart: thermalStart, thermalEnd: ProcessInfo.processInfo.thermalState.rawValue,
            iOSVersion: ProcessInfo.processInfo.operatingSystemVersionString, rowsCSV: csv
        )
    }
}
