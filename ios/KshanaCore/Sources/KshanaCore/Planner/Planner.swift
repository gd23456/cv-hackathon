import Foundation

public enum Budget: String, CaseIterable, Sendable, Identifiable {
    case auto = "Auto"
    case fast = "Fast"
    case balanced = "Balanced"
    case best = "Best"
    public var id: String { rawValue }
}

public struct DeviceState: Sendable {
    public var thermal: ProcessInfo.ThermalState
    public var lowPower: Bool
    public var availableMB: Double

    public init(thermal: ProcessInfo.ThermalState, lowPower: Bool, availableMB: Double) {
        self.thermal = thermal
        self.lowPower = lowPower
        self.availableMB = availableMB
    }

    public static var current: DeviceState {
        DeviceState(
            thermal: ProcessInfo.processInfo.thermalState,
            lowPower: ProcessInfo.processInfo.isLowPowerModeEnabled,
            availableMB: Double(MemoryProbe.availableBytes() ?? UInt64.max / 2) / 1_048_576
        )
    }
}

public struct PlanDecision: Sendable, Equatable {
    public let deadlineMs: Double
    public let maxCandidates: Int
    public let useVerifier: Bool
    public let postProcess: Bool
    public let reason: String
}

/// Turns the requested budget, prompt difficulty and live device state into a plan.
/// Extra compute goes to more candidates, which is the thing tested in experiment E1.
public enum Planner {
    static let minimumMemoryForVerifierMB = 400.0

    public static func decide(budget: Budget, complexity: Int, device: DeviceState) -> PlanDecision {
        var deadline: Double
        var maxCandidates: Int
        var reasons: [String]

        switch budget {
        case .fast:
            (deadline, maxCandidates, reasons) = (0, 1, ["Fast: one image"])
        case .balanced:
            (deadline, maxCandidates, reasons) = (1500, 16, ["Balanced: search for up to 1.5 s"])
        case .best:
            (deadline, maxCandidates, reasons) = (5000, 48, ["Best: search for up to 5 s"])
        case .auto:
            if complexity >= 3 {
                (deadline, maxCandidates, reasons) = (3000, 32, ["Complex prompt: search for up to 3 s"])
            } else {
                (deadline, maxCandidates, reasons) = (1000, 10, ["Simple prompt: search for up to 1 s"])
            }
        }

        var useVerifier = budget != .fast
        var postProcess = true

        if device.lowPower, maxCandidates > 1 {
            deadline /= 2
            maxCandidates = max(1, maxCandidates / 2)
            reasons.append("Low Power Mode: half the search")
        }
        switch device.thermal {
        case .serious:
            if maxCandidates > 2 { reasons.append("Device is hot: at most 2 images") }
            maxCandidates = min(maxCandidates, 2)
        case .critical:
            maxCandidates = 1
            useVerifier = false
            postProcess = false
            reasons.append("Device is very hot: one image, no checks")
        default:
            break
        }
        if device.availableMB < minimumMemoryForVerifierMB, useVerifier {
            useVerifier = false
            maxCandidates = 1
            reasons.append("Low memory: checks off")
        }
        return PlanDecision(
            deadlineMs: deadline, maxCandidates: maxCandidates, useVerifier: useVerifier,
            postProcess: postProcess, reason: reasons.joined(separator: ". ")
        )
    }

    /// Ranks a candidate: mostly checklist facts, then prompt alignment, minus artifacts.
    /// With nothing checkable, alignment alone decides.
    public static func score(checks: [CheckResult], alignment: Float, quality: QualityReport) -> Double {
        let verifiable = checks.filter { $0.state != .notVerifiable }
        // MobileCLIP cosines for matching images sit roughly in this range.
        let aligned = max(0, min(1, (Double(alignment) - 0.1) / 0.3))
        let base: Double
        if verifiable.isEmpty {
            base = aligned
        } else {
            let passed = Double(verifiable.filter { $0.state == .passed }.count) / Double(verifiable.count)
            base = 0.7 * passed + 0.3 * aligned
        }
        return base - quality.penalty
    }
}

/// Stops prompts we won't generate. This is a simple word list with known limits,
/// stated openly in the documentation: it is not a complete safety system.
public enum PromptSafety {
    private static let blocked: Set<String> = [
        "nude", "nudes", "naked", "porn", "porno", "pornographic", "nsfw", "explicit", "sexual", "sex",
        "topless", "erotic", "hentai", "genitals", "rape", "gore", "beheading", "decapitated", "dismembered",
    ]

    public static func isBlocked(_ prompt: String) -> Bool {
        let words = prompt.lowercased().split { !$0.isLetter }
        return words.contains { blocked.contains(String($0)) }
    }
}
