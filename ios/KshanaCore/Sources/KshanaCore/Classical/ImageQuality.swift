import Foundation

/// Classical no-reference measurements of a generated image.
public struct QualityReport: Sendable, Equatable {
    /// Variance of the Laplacian of luma: low means blurry.
    public let sharpness: Double
    /// Standard deviation of luma in 0...1.
    public let contrast: Double
    /// Hasler-Suesstrunk colourfulness.
    public let colorfulness: Double
    /// Fraction of pixels with all channels clipped near black or white.
    public let clippedFraction: Double
    /// Strength of an 8-pixel grid, which latent decoders can leave behind.
    public let gridScore: Double
    /// Largest difference between a channel mean and the overall mean, in 0...1.
    public let colorCast: Double

    /// A flat frame means the generator failed outright, so another seed is needed.
    public var isBlank: Bool { contrast < 0.02 }
    public var hasGridArtifact: Bool { gridScore > 0.35 }
    public var isLowContrast: Bool { contrast < 0.12 }
    public var isClipped: Bool { clippedFraction > 0.25 }

    /// Penalty applied when ranking candidates.
    public var penalty: Double {
        var total = 0.0
        if hasGridArtifact { total += 0.1 }
        if isClipped { total += 0.1 }
        if sharpness < 0.0003 { total += 0.1 }
        return total
    }
}

public enum ImageQuality {
    public static func measure(_ image: RGBAImage) -> QualityReport {
        let (w, h) = (image.width, image.height)
        let luma = Luma.extract(image)
        let n = Double(w * h)

        var sum = 0.0, squares = 0.0
        for value in luma { sum += Double(value); squares += Double(value * value) }
        let mean = sum / n
        let contrast = max(0, squares / n - mean * mean).squareRoot()

        var lapSum = 0.0, lapSquares = 0.0, lapCount = 0.0
        if w > 2, h > 2 {
            for y in 1..<(h - 1) {
                for x in 1..<(w - 1) {
                    let i = y * w + x
                    let lap = Double(luma[i - 1] + luma[i + 1] + luma[i - w] + luma[i + w] - 4 * luma[i])
                    lapSum += lap
                    lapSquares += lap * lap
                    lapCount += 1
                }
            }
        }
        let sharpness = lapCount > 0 ? lapSquares / lapCount - (lapSum / lapCount) * (lapSum / lapCount) : 0

        var rSum = 0.0, gSum = 0.0, bSum = 0.0
        var rgSum = 0.0, rgSquares = 0.0, ybSum = 0.0, ybSquares = 0.0, clipped = 0.0
        for i in 0..<(w * h) {
            let r = Double(image.pixels[i * 4]), g = Double(image.pixels[i * 4 + 1]), b = Double(image.pixels[i * 4 + 2])
            rSum += r; gSum += g; bSum += b
            let rg = r - g, yb = 0.5 * (r + g) - b
            rgSum += rg; rgSquares += rg * rg
            ybSum += yb; ybSquares += yb * yb
            if (r > 250 && g > 250 && b > 250) || (r < 5 && g < 5 && b < 5) { clipped += 1 }
        }
        let rgVar = max(0, rgSquares / n - (rgSum / n) * (rgSum / n))
        let ybVar = max(0, ybSquares / n - (ybSum / n) * (ybSum / n))
        let colorfulness = (rgVar + ybVar).squareRoot()
            + 0.3 * ((rgSum / n) * (rgSum / n) + (ybSum / n) * (ybSum / n)).squareRoot()
        let overall = (rSum + gSum + bSum) / (3 * n)
        let cast = max(abs(rSum / n - overall), abs(gSum / n - overall), abs(bSum / n - overall)) / 255

        return QualityReport(
            sharpness: sharpness, contrast: contrast, colorfulness: colorfulness,
            clippedFraction: clipped / n, gridScore: gridScore(luma, width: w, height: h), colorCast: cast
        )
    }

    /// Mean horizontal gradient by column position modulo 8, using a single
    /// DFT bin at the period-8 frequency relative to overall gradient energy.
    static func gridScore(_ luma: [Float], width: Int, height: Int, period: Int = 8) -> Double {
        guard width > period * 2, height > 1 else { return 0 }
        var buckets = [Double](repeating: 0, count: period)
        var counts = [Double](repeating: 0, count: period)
        for y in stride(from: 0, to: height, by: 2) {
            for x in 1..<width {
                let i = y * width + x
                buckets[x % period] += Double(abs(luma[i] - luma[i - 1]))
                counts[x % period] += 1
            }
        }
        let means = zip(buckets, counts).map { $1 > 0 ? $0 / $1 : 0 }
        let overall = means.reduce(0, +) / Double(period)
        guard overall > 1e-6 else { return 0 }
        var real = 0.0, imaginary = 0.0
        for (k, value) in means.enumerated() {
            let angle = 2 * Double.pi * Double(k) / Double(period)
            real += value * cos(angle)
            imaginary += value * sin(angle)
        }
        return (real * real + imaginary * imaginary).squareRoot() / (Double(period) * overall)
    }
}

enum Luma {
    /// Rec. 601 luma in 0...1.
    static func extract(_ image: RGBAImage) -> [Float] {
        var out = [Float](repeating: 0, count: image.width * image.height)
        for i in 0..<out.count {
            let r = Float(image.pixels[i * 4]), g = Float(image.pixels[i * 4 + 1]), b = Float(image.pixels[i * 4 + 2])
            out[i] = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        }
        return out
    }

    /// Rescales RGB by the luma ratio, which preserves hue while changing brightness.
    static func apply(_ newLuma: [Float], from oldLuma: [Float], to image: RGBAImage) -> RGBAImage {
        var pixels = image.pixels
        for i in 0..<newLuma.count {
            let ratio = oldLuma[i] > 0.004 ? newLuma[i] / oldLuma[i] : 1
            for c in 0..<3 {
                pixels[i * 4 + c] = UInt8(max(0, min(255, Float(image.pixels[i * 4 + c]) * ratio + 0.5)))
            }
        }
        return RGBAImage(width: image.width, height: image.height, pixels: pixels)
    }
}
