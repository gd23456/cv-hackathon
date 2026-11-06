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

    private func refreshMemory() {
        if let snapshot = MemoryProbe.snapshot() {
            footprintMB = Double(snapshot.footprintBytes) / 1_048_576
            peakMB = Double(snapshot.lifetimePeakBytes) / 1_048_576
        }
        availableMB = Double(MemoryProbe.availableBytes() ?? 0) / 1_048_576
    }
}
