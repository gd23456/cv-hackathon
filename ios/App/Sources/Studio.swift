import Foundation
import KshanaCore
import Observation
import os
import UIKit

/// App state and the single seam between the UI and the generation engine.
@Observable
final class Studio {
    enum Phase: Equatable {
        case preparing(String)
        case ready
        case generating
        case failed(String)
    }

    private(set) var phase: Phase = .preparing("Starting")
    private(set) var image: UIImage?
    private(set) var timings: GenerationTimings?
    private(set) var loadReport: LoadReport?
    private(set) var footprintMB: Double = 0
    private(set) var peakMB: Double = 0
    private(set) var availableMB: Double = 0
    var prompt = "a red apple to the left of a blue cup on a wooden table"

    private let logger = Logger(subsystem: "com.gourabdash.kshana", category: "studio")
    private let generator: Generator?
    private var nextSeed: UInt64 = 42

    init() {
        if let models = Bundle.main.url(forResource: "Models", withExtension: nil),
           let tokenizer = Bundle.main.url(forResource: "Tokenizer", withExtension: nil) {
            generator = Generator(location: ModelLocation(models: models, tokenizer: tokenizer))
        } else {
            generator = nil
        }
    }

    func prepare() async {
        guard let generator else {
            phase = .failed("The models are missing from this build.")
            return
        }
        refreshMemory()
        logger.notice("available before load: \(self.availableMB, format: .fixed(precision: 0)) MB")
        do {
            let report = try await generator.load { [weak self] stage in
                Task { @MainActor in self?.phase = .preparing(stage) }
            }
            loadReport = report
            refreshMemory()
            logger.notice("loaded in \(report.totalMs, format: .fixed(precision: 0)) ms, footprint \(self.footprintMB, format: .fixed(precision: 0)) MB")
            phase = .ready
        } catch {
            logger.error("load failed: \(String(describing: error), privacy: .public)")
            phase = .failed("Could not load the models: \(error)")
        }
    }

    func generate() async {
        guard let generator, phase == .ready else { return }
        phase = .generating
        let seed = nextSeed
        nextSeed &+= 1
        do {
            let candidate = try await generator.generate(prompt: prompt, seed: seed)
            image = candidate.image.cgImage.map(UIImage.init(cgImage:))
            timings = candidate.timings
            refreshMemory()
            let t = candidate.timings
            logger.notice("seed \(seed): text \(t.textMs, format: .fixed(precision: 1)) unet \(t.unetMs, format: .fixed(precision: 1)) decode \(t.decodeMs, format: .fixed(precision: 1)) footprint \(self.footprintMB, format: .fixed(precision: 0)) peak \(self.peakMB, format: .fixed(precision: 0))")
            phase = .ready
        } catch {
            logger.error("generate failed: \(String(describing: error), privacy: .public)")
            phase = .failed("Generation failed: \(error)")
        }
    }

    /// Measures this device end to end and saves results to Documents:
    /// `bench.txt`, plus the images, for retrieval over USB.
    func runBenchmark(runs: Int = 12) async {
        guard let generator else { return }
        let documents = URL.documentsDirectory
        let prompts = [
            "a red apple to the left of a blue cup on a wooden table",
            "a close-up picture of an old man standing in the rain",
            "a golden retriever puppy sitting on green grass, sunny day",
        ]
        var report = ""
        func log(_ line: String) {
            report += line + "\n"
            FileHandle.standardOutput.write(Data((line + "\n").utf8))
        }

        let version = ProcessInfo.processInfo.operatingSystemVersionString
        log("device iOS \(version), RAM \(ProcessInfo.processInfo.physicalMemory / 1_048_576) MB, thermal \(ProcessInfo.processInfo.thermalState.rawValue), low power \(ProcessInfo.processInfo.isLowPowerModeEnabled)")
        if let load = loadReport {
            log(String(format: "load ms: text %.0f, unet %.0f, decoder %.0f, total %.0f", load.textMs, load.unetMs, load.decoderMs, load.totalMs))
        }
        refreshMemory()
        log(String(format: "after load MB: footprint %.0f, peak %.0f, available %.0f", footprintMB, peakMB, availableMB))

        for run in 0..<runs {
            do {
                let prompt = prompts[run % prompts.count]
                let candidate = try await generator.generate(prompt: prompt, seed: 1000 + UInt64(run))
                let t = candidate.timings
                refreshMemory()
                log(String(format: "run %d ms: text %.1f, unet %.1f, decode %.1f, total %.1f | MB footprint %.0f, peak %.0f, available %.0f | thermal %d", run, t.textMs, t.unetMs, t.decodeMs, t.totalMs, footprintMB, peakMB, availableMB, ProcessInfo.processInfo.thermalState.rawValue))
                if run < prompts.count {
                    try candidate.image.writePNG(to: documents.appending(path: "bench_\(run).png"))
                }
            } catch {
                log("run \(run) FAILED: \(error)")
            }
        }
        try? report.write(to: documents.appending(path: "bench.txt"), atomically: true, encoding: .utf8)
        log("benchmark done")
    }

    /// Prints where Core ML plans to run each model on this device.
    func runPlanReport() async {
        guard let models = Bundle.main.url(forResource: "Models", withExtension: nil) else { return }
        func log(_ line: String) { FileHandle.standardOutput.write(Data((line + "\n").utf8)) }
        for name in ["TAESD", "TextEncoder", "UNet"] {
            log("planning \(name)")
            do {
                let plan = try await ComputePlanReport.load(modelAt: models.appending(path: "\(name).mlmodelc"))
                let onCPU = plan.cpuOpNames.sorted { $0.value > $1.value }.prefix(6).map { "\($0.key) x\($0.value)" }.joined(separator: ", ")
                log("plan \(name): NE \(plan.neuralEngineOps), GPU \(plan.gpuOps), CPU \(plan.cpuOps), unknown \(plan.unknownOps), of \(plan.totalOps) | CPU ops: \(onCPU)")
            } catch {
                log("plan \(name) FAILED: \(error)")
            }
        }
        log("plan done")
    }

    private func refreshMemory() {
        if let snapshot = MemoryProbe.snapshot() {
            footprintMB = Double(snapshot.footprintBytes) / 1_048_576
            peakMB = Double(snapshot.lifetimePeakBytes) / 1_048_576
        }
        availableMB = Double(MemoryProbe.availableBytes() ?? 0) / 1_048_576
    }
}
