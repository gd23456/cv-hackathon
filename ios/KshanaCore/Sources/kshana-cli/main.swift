import CoreML
import Foundation
import KshanaCore

// Runs the same engine as the app on this Mac, for validation and benchmarks:
//   kshana-cli --models DIR --tokenizer DIR --prompt TEXT --seed N --out FILE.png [--runs N] [--units ne|gpu|all]
func option(_ name: String) -> String? {
    let args = CommandLine.arguments
    guard let index = args.firstIndex(of: "--\(name)"), index + 1 < args.count else { return nil }
    return args[index + 1]
}

guard let models = option("models"), let tokenizer = option("tokenizer"), let prompt = option("prompt") else {
    print("usage: kshana-cli --models DIR --tokenizer DIR --prompt TEXT [--seed N] [--out FILE.png] [--runs N] [--units ne|gpu|all]")
    exit(2)
}

let units: MLComputeUnits = switch option("units") ?? "ne" {
case "gpu": .cpuAndGPU
case "all": .all
default: .cpuAndNeuralEngine
}

let generator = Generator(
    location: ModelLocation(models: URL(filePath: models), tokenizer: URL(filePath: tokenizer)),
    computeUnits: units
)

let report = try await generator.load()
print(String(format: "load ms: text %.0f, unet %.0f, decoder %.0f", report.textMs, report.unetMs, report.decoderMs))

let seed = UInt64(option("seed") ?? "42") ?? 42
let runs = Int(option("runs") ?? "1") ?? 1
var latest: GeneratedCandidate?
for run in 0..<runs {
    let candidate = try await generator.generate(prompt: prompt, seed: seed &+ UInt64(run))
    let t = candidate.timings
    print(String(format: "run %d ms: text %.1f, unet %.1f, decode %.1f, total %.1f", run, t.textMs, t.unetMs, t.decodeMs, t.totalMs))
    latest = candidate
}

if let out = option("out"), let latest {
    try latest.image.writePNG(to: URL(filePath: out))
    print("wrote \(out)")
}
if let memory = MemoryProbe.snapshot() {
    print(String(format: "footprint MB: now %.0f, peak %.0f", Double(memory.footprintBytes) / 1e6, Double(memory.lifetimePeakBytes) / 1e6))
}
