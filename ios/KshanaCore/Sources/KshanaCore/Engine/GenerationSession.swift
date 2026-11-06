import Foundation

public struct CandidateResult: Sendable {
    public let index: Int
    public let seed: UInt64
    public let image: RGBAImage
    public let score: Double
    public let checks: [CheckResult]
    public let alignment: Float?
    public let quality: QualityReport
    public let timings: GenerationTimings
    public let verifyMs: Double
    public let isBest: Bool
}

public struct SessionSummary: Sendable {
    public let best: CandidateResult
    public let finalImage: RGBAImage
    public let stages: [ClassicalStage]
    public let candidatesTried: Int
    public let elapsedMs: Double
    public let firstImageMs: Double
    public let safetyRejections: Int
    public let wasTruncated: Bool
}

public enum SessionEvent: Sendable {
    case planned(ParsedPrompt, PlanDecision)
    case candidate(CandidateResult)
    case finished(SessionSummary)
    case blocked(String)
    case failed(String)
}

/// The anytime loop: show the first image at once, then keep trying seeds until
/// the plan's deadline, publishing each image that beats the current best.
public actor GenerationSession {
    static let refusal = "This image can't be shown. Try describing something different."
    static let maxSafetyRejections = 3
    private let generator: Generator
    private let verifier: Verifier?
    private let safety: SafetyGate

    public init(generator: Generator, verifier: Verifier?, safety: SafetyGate) {
        self.generator = generator
        self.verifier = verifier
        self.safety = safety
    }

    public nonisolated func run(prompt: String, budget: Budget, baseSeed: UInt64) -> AsyncStream<SessionEvent> {
        AsyncStream { continuation in
            let task = Task {
                await self.execute(prompt: prompt, budget: budget, baseSeed: baseSeed, continuation: continuation)
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func execute(prompt: String, budget: Budget, baseSeed: UInt64, continuation: AsyncStream<SessionEvent>.Continuation) async {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            continuation.yield(.failed("Type a description first."))
            return
        }
        guard !PromptSafety.isUnreadable(trimmed) else {
            continuation.yield(.blocked("Please describe the image in English, so the safety filter can read it."))
            return
        }
        guard !PromptSafety.isBlocked(trimmed) else {
            continuation.yield(.blocked("This app doesn't generate that kind of image. Try a different description."))
            return
        }
        guard await safety.isLoaded else {
            continuation.yield(.failed("The safety check isn't ready, so nothing can be shown yet. Close Kshana and try again."))
            return
        }

        do {
            let parsed = PromptParser.parse(trimmed)
            let decision = Planner.decide(budget: budget, complexity: parsed.complexity, device: .current)
            continuation.yield(.planned(parsed, decision))
            let tokens = try await generator.tokenize(trimmed)

            let verifierReady = await verifier?.isLoaded ?? false
            let useVerifier = decision.useVerifier && verifierReady
            let start = ContinuousClock.now
            func elapsed() -> Double {
                let parts = start.duration(to: .now).components
                return Double(parts.seconds) * 1000 + Double(parts.attoseconds) / 1e15
            }

            var best: CandidateResult?
            var firstImageMs: Double?
            var index = 0
            var failures = 0
            var safetyRejections = 0

            while index < decision.maxCandidates, failures < 4 {
                try Task.checkCancellation()
                if best != nil, elapsed() >= decision.deadlineMs { break }

                let seed = baseSeed &+ UInt64(index + failures)
                let generated = try await generator.generate(prompt: trimmed, seed: seed)
                let quality = ImageQuality.measure(generated.image)
                if quality.isBlank {
                    failures += 1
                    continue
                }

                // Nothing leaves this loop for the screen until the gate has passed it.
                guard await safety.allows(generated.image) else {
                    safetyRejections += 1
                    if safetyRejections >= Self.maxSafetyRejections {
                        continuation.yield(.blocked(Self.refusal))
                        return
                    }
                    failures += 1
                    continue
                }

                var checks: [CheckResult] = []
                var alignment: Float?
                var verifyMs = 0.0
                if useVerifier, let verifier {
                    let verification = try await verifier.verify(generated.image, prompt: trimmed, parsed: parsed)
                    checks = verification.checks
                    alignment = verification.alignment
                    verifyMs = verification.detectMs + verification.alignMs
                }

                let score = Planner.score(checks: checks, alignment: alignment ?? 0, quality: quality)
                let isBest = best.map { score > $0.score } ?? true
                let candidate = CandidateResult(
                    index: index, seed: seed, image: generated.image, score: score, checks: checks,
                    alignment: alignment, quality: quality, timings: generated.timings,
                    verifyMs: verifyMs, isBest: isBest
                )
                if isBest { best = candidate }
                if firstImageMs == nil { firstImageMs = elapsed() }
                continuation.yield(.candidate(candidate))
                index += 1

                // Everything checkable passed, so more searching can't improve the facts.
                let verifiable = checks.filter { $0.state != .notVerifiable }
                if !verifiable.isEmpty, verifiable.allSatisfy({ $0.state == .passed }) { break }
            }

            guard let best else {
                continuation.yield(.failed("No usable image came out. Try again, or describe something different."))
                return
            }

            let stages = decision.postProcess
                ? ClassicalPipeline.process(best.image, quality: best.quality, prompt: parsed)
                : []
            let final = stages.last?.image ?? best.image
            guard await safety.allows(final) else {
                continuation.yield(.blocked(Self.refusal))
                return
            }
            continuation.yield(.finished(SessionSummary(
                best: best, finalImage: final, stages: stages, candidatesTried: index,
                elapsedMs: elapsed(), firstImageMs: firstImageMs ?? 0,
                safetyRejections: safetyRejections, wasTruncated: tokens.wasTruncated
            )))
        } catch is CancellationError {
            return
        } catch {
            continuation.yield(.failed("Generation failed: \(error)"))
        }
    }
}
