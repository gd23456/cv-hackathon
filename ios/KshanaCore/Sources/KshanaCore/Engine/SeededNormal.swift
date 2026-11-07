import Foundation

/// Deterministic standard-normal samples: SplitMix64 + Box-Muller.
/// Simple enough to reimplement exactly in Python, so a seed gives the same
/// starting latent on the iPhone, the Mac and in the evaluation scripts.
public struct SeededNormal: Sendable {
    private var state: UInt64

    public init(seed: UInt64) {
        state = seed
    }

    private mutating func nextUInt64() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform in (0, 1], so the logarithm below is always finite.
    private mutating func nextUnit() -> Double {
        (Double(nextUInt64() >> 11) + 1) / 9_007_199_254_740_992
    }

    public mutating func fill(count: Int) -> [Float] {
        var out = [Float](repeating: 0, count: count)
        var i = 0
        while i < count {
            let u1 = nextUnit()
            let u2 = nextUnit()
            let radius = (-2 * log(u1)).squareRoot()
            let angle = 2 * Double.pi * u2
            out[i] = Float(radius * cos(angle))
            if i + 1 < count { out[i + 1] = Float(radius * sin(angle)) }
            i += 2
        }
        return out
    }

    /// Spherical interpolation between two noise vectors. Unlike a straight blend it keeps
    /// the noise at the strength the model expects, so every point gives a real image.
    public static func slerp(_ a: [Float], _ b: [Float], t: Float) -> [Float] {
        var dot: Float = 0, normA: Float = 0, normB: Float = 0
        for i in a.indices {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        let cosine = max(-1, min(1, dot / (normA.squareRoot() * normB.squareRoot())))
        let theta = acos(cosine)
        guard theta > 1e-4 else { return a }
        let wa = sin((1 - t) * theta) / sin(theta)
        let wb = sin(t * theta) / sin(theta)
        return a.indices.map { wa * a[$0] + wb * b[$0] }
    }
}
