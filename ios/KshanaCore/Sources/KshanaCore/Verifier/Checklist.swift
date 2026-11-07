import CoreGraphics
import Foundation

/// One detected object. The box is normalized, with the origin at the top left.
public struct Detection: Sendable, Equatable {
    public let label: String
    public let confidence: Float
    public let box: CGRect

    public init(label: String, confidence: Float, box: CGRect) {
        self.label = label
        self.confidence = confidence
        self.box = box
    }
}

public struct CheckResult: Sendable, Identifiable, Equatable {
    public enum State: Sendable, Equatable { case passed, failed, notVerifiable }
    public let title: String
    public let state: State
    public var id: String { title }
}

/// Scores a generated image against the facts parsed from its prompt.
/// Thresholds follow GenEval's rules, with the count threshold lowered because
/// a tiny on-device detector rarely reaches GenEval's 0.9.
public enum Checklist {
    static let presenceConfidence: Float = 0.25
    static let countConfidence: Float = 0.5
    static let positionMargin: CGFloat = 0.1
    static let colorFraction = 0.35
    /// Above this share of another colour named in the prompt, the colour has leaked onto this object.
    static let leakFraction = 0.2
    /// Boxes overlapping more than this aren't really beside each other.
    static let maxRelationOverlap: CGFloat = 0.3

    public static func evaluate(_ prompt: ParsedPrompt, detections: [Detection], image: RGBAImage) -> [CheckResult] {
        var results: [CheckResult] = []

        func matches(_ object: ParsedObject) -> [Detection] {
            detections.filter { $0.label == object.label && $0.confidence >= presenceConfidence }
        }

        for object in prompt.objects {
            guard object.label != nil else {
                results.append(CheckResult(title: object.noun, state: .notVerifiable))
                continue
            }
            let found = matches(object)
            if let count = object.count {
                let confident = found.filter { $0.confidence >= countConfidence }.count
                results.append(CheckResult(title: "\(count) \(object.noun)", state: confident == count ? .passed : .failed))
            } else {
                results.append(CheckResult(title: object.noun, state: found.isEmpty ? .failed : .passed))
            }
            if let color = object.color {
                let others = Set(prompt.objects.compactMap(\.color)).subtracting([color])
                let passed = found.contains { detection in
                    colorShare(of: color, in: detection.box, image: image) >= colorFraction
                        && others.allSatisfy { colorShare(of: $0, in: detection.box, image: image) < leakFraction }
                }
                results.append(CheckResult(title: "\(color) \(object.noun)", state: passed ? .passed : .failed))
            }
        }

        for relation in prompt.relations {
            let subject = prompt.objects[relation.subject]
            let object = prompt.objects[relation.object]
            let title = "\(subject.noun) \(relation.relation.rawValue) \(object.noun)"
            guard subject.label != nil, object.label != nil else {
                results.append(CheckResult(title: title, state: .notVerifiable))
                continue
            }
            let holds = matches(subject).contains { a in
                matches(object).contains { b in satisfies(relation.relation, a.box, b.box) }
            }
            results.append(CheckResult(title: title, state: holds ? .passed : .failed))
        }
        return results
    }

    static func overlap(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let shared = a.intersection(b)
        guard !shared.isNull else { return 0 }
        let area = shared.width * shared.height
        let smaller = min(a.width * a.height, b.width * b.height)
        return smaller > 0 ? area / smaller : 0
    }

    static func satisfies(_ relation: SpatialRelation, _ a: CGRect, _ b: CGRect) -> Bool {
        guard overlap(a, b) <= maxRelationOverlap else { return false }
        return switch relation {
        case .leftOf: a.midX + positionMargin < b.midX
        case .rightOf: a.midX > b.midX + positionMargin
        case .above: a.midY + positionMargin < b.midY
        case .below: a.midY > b.midY + positionMargin
        }
    }

    /// Fraction of pixels in the centre of the box matching a named colour, in HSV.
    static func colorShare(of color: String, in box: CGRect, image: RGBAImage) -> Double {
        let w = CGFloat(image.width), h = CGFloat(image.height)
        let inset = box.insetBy(dx: box.width * 0.2, dy: box.height * 0.2)
        let x0 = max(0, Int(inset.minX * w)), x1 = min(image.width, Int(inset.maxX * w))
        let y0 = max(0, Int(inset.minY * h)), y1 = min(image.height, Int(inset.maxY * h))
        guard x1 > x0, y1 > y0 else { return 0 }
        var hits = 0, total = 0
        for y in stride(from: y0, to: y1, by: 2) {
            for x in stride(from: x0, to: x1, by: 2) {
                let i = (y * image.width + x) * 4
                let hsv = hsv(image.pixels[i], image.pixels[i + 1], image.pixels[i + 2])
                if isColor(color, hsv.h, hsv.s, hsv.v) { hits += 1 }
                total += 1
            }
        }
        return Double(hits) / Double(total)
    }

    static func hsv(_ r: UInt8, _ g: UInt8, _ b: UInt8) -> (h: Double, s: Double, v: Double) {
        let r = Double(r) / 255, g = Double(g) / 255, b = Double(b) / 255
        let maxC = max(r, g, b), minC = min(r, g, b), delta = maxC - minC
        var hue = 0.0
        if delta > 0 {
            if maxC == r { hue = (g - b) / delta }
            else if maxC == g { hue = (b - r) / delta + 2 }
            else { hue = (r - g) / delta + 4 }
            hue *= 60
            if hue < 0 { hue += 360 }
        }
        return (hue, maxC == 0 ? 0 : delta / maxC, maxC)
    }

    static func isColor(_ name: String, _ h: Double, _ s: Double, _ v: Double) -> Bool {
        let chromatic = s > 0.25 && v > 0.2
        return switch name {
        case "black": v < 0.2
        case "white": s < 0.15 && v > 0.8
        case "gray": s < 0.15 && v >= 0.2 && v <= 0.8
        case "brown": chromatic && h >= 10 && h < 45 && v < 0.65
        case "red": chromatic && (h >= 345 || h < 15)
        case "orange": chromatic && h >= 15 && h < 40 && v >= 0.65
        case "yellow": chromatic && h >= 40 && h < 70
        case "green": chromatic && h >= 70 && h < 165
        case "blue": chromatic && h >= 165 && h < 255
        case "purple": chromatic && h >= 255 && h < 300
        case "pink": chromatic && h >= 300 && h < 345
        default: false
        }
    }
}
