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

/// Stops prompts we won't generate, before any image is made. Deliberately strict,
/// because this app is shown live. It's a filter with known limits, not a guarantee:
/// the image safety gate is the second line of defence.
public enum PromptSafety {
    private static let sexual = "nudes?|naked|nudity|nudist|topless|bottomless|unclothed|nsfw|porn\\w*|hentai|erotic\\w*|xxx|"
        + "sex|sexy|sexual\\w*|genitals?|genitalia|penis(?:es)?|cocks?|dicks?|phallus|phallic|"
        + "vaginas?|vulvas?|puss(?:y|ies)|clit\\w*|breasts?|boob(?:s|ies)?|tits?|titt(?:y|ies)|nipples?|areolas?|"
        + "butts?|buttocks?|ass|asses|arse|booty|anus|anal|crotch|groin|testicles?|semen|cum|orgasm\\w*|masturbat\\w*|"
        + "blowjobs?|handjobs?|intercourse|fetish\\w*|bdsm|orgy|orgies|lingerie|thongs?|bikinis?|underwear|panties|"
        + "undress\\w*|stripper|striptease|seductive|provocative|sensual|"
        + "nangi|nanga|chut|choot|lund|gaand|gand|chuchi|randi|bhosd\\w*|"
        + "private parts|in the buff|no garments|doing it|adult film|exposed body|uncovered chest|no clothes|without clothes|wearing nothing|bare skin|bare chest"
    private static let harm = "gore|gory|beheaded|beheading|decapitated|decapitation|dismembered|dismemberment|disembowell?ed|"
        + "mutilated|mutilation|torture|tortured|massacre|corpses?|lynching|lynched|"
        + "suicide|suicidal|self harm|slit wrists?|nazi|swastika|kkk|white power"
    private static let minor = "child|children|kid|kids|minor|minors|underage|toddler|infant|baby|babies|teen|teens|teenager|"
        + "boy|boys|girl|girls|schoolgirl|schoolboy|loli|shota"
    private static let minorContext = "swimsuit|bedroom|bed|shirtless|bath|bathing|shower|kiss\\w*"
    private static let harmlessPhrases = "\\b(?:chicken|turkey|duck) breasts?\\b|\\b(?:cigarette|rifle) butts?\\b"
    private static let insideWord = ["penis", "vagina", "pussy", "porn", "hentai", "blowjob", "handjob", "orgasm", "boobs", "titties", "masturbat"]

    private static func word(_ pattern: String) -> NSRegularExpression {
        // The patterns above are constants, so a failure here is a programming mistake.
        try! NSRegularExpression(pattern: "\\b(?:\(pattern))\\b")
    }
    private static let sexualRegex = word(sexual)
    private static let harmRegex = word(harm)
    private static let minorRegex = word(minor)
    private static let minorContextRegex = word(minorContext)
    private static let harmlessRegex = try! NSRegularExpression(pattern: harmlessPhrases)
    private static let spacedRegex = try! NSRegularExpression(pattern: "\\b(?:[a-z] +){2,}[a-z]\\b")

    /// Lowercases, folds accents, undoes number and symbol swaps, and joins spaced-out letters.
    static func normalized(_ text: String) -> String {
        let swaps: [Character: Character] = ["0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"]
        let folded = text.folding(options: [.diacriticInsensitive, .widthInsensitive], locale: nil).lowercased()
        let plain = String(folded.map { character -> Character in
            let swapped = swaps[character] ?? character
            return ("a"..."z").contains(swapped) || swapped.isWhitespace ? swapped : " "
        })
        let range = NSRange(plain.startIndex..., in: plain)
        var joined = plain
        for match in spacedRegex.matches(in: plain, range: range).reversed() {
            guard let matched = Range(match.range, in: joined) else { continue }
            joined.replaceSubrange(matched, with: joined[matched].replacing(" ", with: ""))
        }
        return joined.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }

    /// Mostly non-Latin text can't be read by this filter, so it's refused rather than trusted.
    public static func isUnreadable(_ prompt: String) -> Bool {
        let letters = prompt.filter(\.isLetter)
        guard !letters.isEmpty else { return true }
        let latin = letters.filter { $0.isASCII || $0.unicodeScalars.first.map { $0.value < 0x250 } == true }
        return Double(latin.count) / Double(letters.count) < 0.7
    }

    public static func isBlocked(_ prompt: String) -> Bool {
        let text = normalized(prompt)
        let cleaned = harmlessRegex.stringByReplacingMatches(in: text, range: NSRange(text.startIndex..., in: text), withTemplate: " ")
        let range = NSRange(cleaned.startIndex..., in: cleaned)
        func found(_ regex: NSRegularExpression) -> Bool { regex.firstMatch(in: cleaned, range: range) != nil }
        if found(sexualRegex) || found(harmRegex) { return true }
        if cleaned.split(separator: " ").contains(where: { word in insideWord.contains { word.contains($0) } }) { return true }
        return found(minorRegex) && found(minorContextRegex)
    }
}
