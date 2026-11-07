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
    /// Changes whenever a better image replaces the current one, to drive its transition.
    private(set) var imageRevision = 0
    private(set) var flyingCards: [FlyingCard] = []
    private(set) var detections: [Detection] = []
    private(set) var burstCount = 0
    private(set) var exportImage: UIImage?
    private(set) var parsed: ParsedPrompt?
    private(set) var plan: PlanDecision?
    private(set) var checks: [CheckResult] = []
    private(set) var triedCount = 0
    private(set) var latest: CandidateResult?
    private(set) var summary: SessionSummary?
    private(set) var notice: String?
    private(set) var completedRuns = 0
    private(set) var loadReport: LoadReport?
    private(set) var verifierAvailable = false
    private(set) var verifierStatus = "loading"
    private(set) var acceleration: AccelerationReport?
    private(set) var benchmark: BenchmarkResult?
    private(set) var isBenchmarking = false
    private(set) var device = DeviceSnapshot.current

    var prompt = ""
    var budget: Budget = .auto

    private let logger = Logger(subsystem: "com.gourabdash.kshana", category: "studio")
    private let generator: Generator?
    private let verifier: Verifier?
    private let safety: SafetyGate?
    private let session: GenerationSession?
    private let textOnCPU = CommandLine.arguments.contains("--text-cpu")
    private var generationTask: Task<Void, Never>?
    private var nextSeed = UInt64(Date.now.timeIntervalSince1970)

    init() {
        let bundle = Bundle.main
        let useCPUForText = CommandLine.arguments.contains("--text-cpu")
        guard
            let models = bundle.url(forResource: "Models", withExtension: nil),
            let tokenizer = bundle.url(forResource: "Tokenizer", withExtension: nil)
        else {
            (generator, verifier, safety, session) = (nil, nil, nil, nil)
            return
        }
        let generator = Generator(
            location: ModelLocation(models: models, tokenizer: tokenizer),
            textComputeUnits: useCPUForText ? .cpuOnly : .cpuAndNeuralEngine
        )
        let verifier = bundle.url(forResource: "Verifier", withExtension: nil).map {
            Verifier(location: VerifierLocation(directory: $0, tokenizer: tokenizer))
        }
        let safety = SafetyGate(modelURL: bundle.url(forResource: "Verifier", withExtension: nil)?.appending(path: "nsfw_marqo.mlmodelc"))
        self.generator = generator
        self.verifier = verifier
        self.safety = safety
        session = GenerationSession(generator: generator, verifier: verifier, safety: safety)
    }

    var canGenerate: Bool {
        phase == .ready && !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    /// Every reinstall gets a new bundle path, and Core ML keeps about 1 GB of compiled
    /// Neural Engine cache per path. Old entries never get reused, so remove them.
    private func clearStaleModelCache() {
        let key = "lastBundlePath"
        let path = Bundle.main.bundlePath
        guard UserDefaults.standard.string(forKey: key) != path else { return }
        if UserDefaults.standard.string(forKey: key) != nil,
           let identifier = Bundle.main.bundleIdentifier {
            try? FileManager.default.removeItem(at: URL.cachesDirectory.appending(path: identifier))
        }
        UserDefaults.standard.set(path, forKey: key)
    }

    func prepare() async {
        guard let generator else {
            phase = .failed("The models are missing from this build.")
            return
        }
        clearStaleModelCache()
        // Compiling models on a nearly full phone can end the app, so check first.
        let free = (try? URL.documentsDirectory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
            .volumeAvailableCapacityForImportantUsage ?? .max
        guard free > 500_000_000 else {
            FileHandle.standardOutput.write(Data("low storage: \(free / 1_000_000) MB free\n".utf8))
            phase = .failed("This iPhone is almost full. Free up at least 500 MB, then open Kshana again.")
            return
        }
        do {
            loadReport = try await generator.load { [weak self] stage in
                Task { @MainActor in self?.phase = .preparing(stage) }
            }
            phase = .preparing("Neural Engine check")
            let acceleration = try await generator.ensureAccelerated()
            self.acceleration = acceleration
            if !acceleration.isAccelerated {
                notice = "The Neural Engine isn't available right now, so images take about a second each. Reopening Kshana may fix this."
            }
            phase = .preparing("Safety check")
            // If this fails the app still opens, but it refuses to show any image.
            do {
                try await safety?.load()
            } catch {
                logger.error("safety gate failed: \(String(describing: error), privacy: .public)")
                notice = "The safety check couldn't load, so images can't be shown. Close Kshana and try again."
            }
            phase = .preparing("Checks")
            await loadVerifier()
            refreshDevice()
            phase = .ready
        } catch {
            logger.error("load failed: \(String(describing: error), privacy: .public)")
            phase = .failed("Couldn't load the models. Close Kshana and try again.")
        }
    }

    /// Neural Engine setup for these models can fail on a first launch, so try twice.
    /// A trial check also warms them up, which would otherwise delay the first real image.
    private func loadVerifier() async {
        guard let generator, let verifier else {
            verifierStatus = "not bundled"
            return
        }
        for attempt in 1...2 {
            do {
                try await verifier.load()
                let probe = try await generator.generate(prompt: "a photo", seed: 3)
                _ = try await verifier.verify(probe.image, prompt: "a photo", parsed: PromptParser.parse("a photo"))
                verifierAvailable = true
                verifierStatus = "on, attempt \(attempt)"
                return
            } catch {
                verifierStatus = "failed: \(error)"
                logger.error("verifier attempt \(attempt): \(String(describing: error), privacy: .public)")
            }
        }
        notice = "Checks are off because their models couldn't load. Images still work."
    }

    func generate() {
        guard canGenerate, let session else { return }
        generationTask?.cancel()
        phase = .generating
        image = nil
        detections = []
        flyingCards = []
        checks = []
        triedCount = 0
        summary = nil
        exportImage = nil
        notice = nil
        plan = nil
        let seed = nextSeed
        nextSeed &+= 1000
        let stream = session.run(prompt: prompt, budget: budget, baseSeed: seed)
        generationTask = Task {
            for await event in stream { handle(event) }
            if phase == .generating { phase = .ready }
            refreshDevice()
        }
    }

    private func handle(_ event: SessionEvent) {
        switch event {
        case .planned(let parsed, let plan):
            self.parsed = parsed
            self.plan = plan
        case .candidate(let candidate):
            triedCount = candidate.index + 1
            latest = candidate
            launchCard(for: candidate)
            if candidate.isBest {
                detections = candidate.detections
                image = candidate.image.cgImage.map(UIImage.init(cgImage:))
                imageRevision += 1
                checks = candidate.checks
            }
        case .finished(let summary):
            self.summary = summary
            image = summary.finalImage.cgImage.map(UIImage.init(cgImage:))
            exportImage = makeExportImage(from: summary)
            checks = summary.best.checks
            completedRuns += 1
            let verifiable = summary.best.checks.filter { $0.state != .notVerifiable }
            if !verifiable.isEmpty, Double(verifiable.filter { $0.state == .passed }.count) / Double(verifiable.count) >= 0.8 {
                burstCount += 1
            }
            if summary.wasTruncated {
                notice = "Only the first 77 tokens of a long description were used."
            }
            phase = .ready
        case .blocked(let message), .failed(let message):
            image = nil
            notice = message
            phase = .ready
        }
    }

    /// Heavy animation is switched off when the phone is hot or saving power, because
    /// the same resources make the images.
    var effectsAllowed: Bool {
        !device.lowPower && device.thermal != .serious && device.thermal != .critical
    }

    private func launchCard(for candidate: CandidateResult) {
        guard effectsAllowed, let cgImage = candidate.thumbnail?.cgImage else { return }
        let now = Date.now
        flyingCards.removeAll { now.timeIntervalSince($0.born) > WarpField.life }
        flyingCards.append(FlyingCard(
            id: Int(truncatingIfNeeded: candidate.seed), image: UIImage(cgImage: cgImage), born: now,
            angle: Double(candidate.index) * 2.399963, spin: candidate.index % 2 == 0 ? 70 : -70
        ))
    }

    /// Runs one real search through the animated interface: `--ui-test`.
    func runInterfaceTest() async {
        prompt = "a red apple to the left of a blue cup on a wooden table"
        budget = .best
        generate()
        await generationTask?.value
        refreshDevice()
        let s = summary
        let line = String(
            format: "ui test: tried %d in %.0f ms, first image %.0f ms, effects %@, thermal %d, footprint %.0f MB, peak %.0f MB\n",
            s?.candidatesTried ?? -1, s?.elapsedMs ?? -1, s?.firstImageMs ?? -1, effectsAllowed ? "on" : "off",
            device.thermal.rawValue, device.footprintMB, device.peakMB
        )
        try? line.write(to: URL.documentsDirectory.appending(path: "ui_test.txt"), atomically: true, encoding: .utf8)
    }

    /// Generation can't continue in the background, so stop cleanly.
    func cancelForBackground() {
        generationTask?.cancel()
        if phase == .generating { phase = .ready }
    }

    func handleMemoryWarning() {
        logger.notice("memory warning: releasing the verifier")
        FileHandle.standardOutput.write(Data("memory warning, available \(DeviceSnapshot.current.availableMB) MB\n".utf8))
        generationTask?.cancel()
        if phase == .generating { phase = .ready }
        verifierAvailable = false
        notice = "Memory is low, so checks were turned off for now."
        Task { await verifier?.unload() }
    }

    /// A larger copy for saving, resampled with Lanczos.
    private func makeExportImage(from summary: SessionSummary) -> UIImage? {
        guard let large = try? ImageOps.resized(summary.finalImage, width: 1024, height: 1024) else { return image }
        return large.cgImage.map(UIImage.init(cgImage:))
    }

    func refreshDevice() {
        device = .current
    }

    // MARK: - Benchmarks

    func runBenchmark() async {
        guard let generator, !isBenchmarking else { return }
        isBenchmarking = true
        defer { isBenchmarking = false }
        benchmark = try? await BenchmarkResult.measure(generator: generator, runs: 20)
        refreshDevice()
    }

    /// Headless run for collecting real numbers over USB: `--bench`.
    func runCommandLineBenchmark() async {
        guard let generator, let session else { return }
        let documents = URL.documentsDirectory
        var report = ""
        func log(_ line: String) {
            report += line + "\n"
            FileHandle.standardOutput.write(Data((line + "\n").utf8))
        }

        let info = ProcessInfo.processInfo
        let freeMB = ((try? documents.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?.volumeAvailableCapacityForImportantUsage ?? 0) / 1_000_000
        log("storage free \(freeMB) MB")
        log("device iOS \(info.operatingSystemVersionString), RAM \(info.physicalMemory / 1_048_576) MB, thermal \(info.thermalState.rawValue), low power \(info.isLowPowerModeEnabled), text-cpu \(textOnCPU)")
        if let load = loadReport {
            log(String(format: "load ms: text %.0f, unet %.0f, decoder %.0f, total %.0f", load.textMs, load.unetMs, load.decoderMs, load.totalMs))
        }
        log("verifier \(verifierStatus)")
        if let acceleration {
            log(String(format: "acceleration: unet %.1f ms, reloads %d, accelerated %@", acceleration.unetMs, acceleration.reloads, acceleration.isAccelerated ? "yes" : "NO"))
        }
        refreshDevice()
        log(String(format: "after load MB: footprint %.0f, peak %.0f, available %.0f", device.footprintMB, device.peakMB, device.availableMB))

        do {
            let quick = CommandLine.arguments.contains("--quick")
            let result = try await BenchmarkResult.measure(generator: generator, runs: quick ? 6 : 12)
            log(result.summaryLine)
        } catch {
            log("generator benchmark FAILED: \(error)")
        }

        let prompts = [
            "a red apple to the left of a blue cup on a wooden table",
            "two dogs sitting on a couch",
            "a golden retriever puppy sitting on green grass, sunny day",
            "a mountain lake at sunset",
        ]
        for (i, text) in prompts.prefix(CommandLine.arguments.contains("--quick") ? 1 : prompts.count).enumerated() {
            for await event in session.run(prompt: text, budget: .best, baseSeed: 5000 + UInt64(i) * 100) {
                switch event {
                case .planned(_, let plan):
                    log("prompt \(i): \(text) | \(plan.reason)")
                case .finished(let s):
                    let checks = s.best.checks.map { "\($0.title)=\($0.state)" }.joined(separator: ", ")
                    log(String(format: "prompt %d: tried %d in %.0f ms, first image %.0f ms, verify %.0f ms, score %.2f, align %.3f, safety rejected %d", i, s.candidatesTried, s.elapsedMs, s.firstImageMs, s.best.verifyMs, s.best.score, s.best.alignment ?? -1, s.safetyRejections))
                    log("  checks: \(checks)")
                    log("  stages: " + s.stages.map { String(format: "%@ %.1f ms %@", $0.name, $0.milliseconds, $0.applied ? "on" : "off") }.joined(separator: ", "))
                    try? s.finalImage.writePNG(to: documents.appending(path: "session_\(i).png"))
                case .blocked(let message), .failed(let message):
                    log("prompt \(i) stopped: \(message)")
                case .candidate:
                    break
                }
            }
        }
        refreshDevice()
        log(String(format: "end MB: footprint %.0f, peak %.0f, available %.0f, thermal %d", device.footprintMB, device.peakMB, device.availableMB, ProcessInfo.processInfo.thermalState.rawValue))
        try? report.write(to: documents.appending(path: "bench.txt"), atomically: true, encoding: .utf8)
        log("benchmark done")
    }

    /// Prints where Core ML plans to run each model on this device: `--plan`.
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
}

struct DeviceSnapshot {
    let footprintMB: Double
    let peakMB: Double
    let availableMB: Double
    let thermal: ProcessInfo.ThermalState
    let lowPower: Bool

    static var current: DeviceSnapshot {
        let memory = MemoryProbe.snapshot()
        return DeviceSnapshot(
            footprintMB: Double(memory?.footprintBytes ?? 0) / 1_048_576,
            peakMB: Double(memory?.lifetimePeakBytes ?? 0) / 1_048_576,
            availableMB: Double(MemoryProbe.availableBytes() ?? 0) / 1_048_576,
            thermal: ProcessInfo.processInfo.thermalState,
            lowPower: ProcessInfo.processInfo.isLowPowerModeEnabled
        )
    }

    var thermalName: String {
        switch thermal {
        case .nominal: "Normal"
        case .fair: "Warm"
        case .serious: "Hot"
        case .critical: "Very hot"
        @unknown default: "Unknown"
        }
    }
}
