import Accelerate
import Foundation

/// Hand-written classical operators, so each can be explained and measured.
public enum ImageOps {
    /// Gray-world white balance: scales channels toward a common mean.
    /// Gains are limited, so a warm sunset doesn't get pushed to neutral.
    public static func grayWorld(_ image: RGBAImage, maxGain: Float = 1.15) -> RGBAImage {
        let count = image.width * image.height
        var sums: [Float] = [0, 0, 0]
        for i in 0..<count {
            for c in 0..<3 { sums[c] += Float(image.pixels[i * 4 + c]) }
        }
        let overall = (sums[0] + sums[1] + sums[2]) / 3
        guard sums.allSatisfy({ $0 > 0 }) else { return image }
        let gains = sums.map { max(1 / maxGain, min(maxGain, overall / $0)) }
        var pixels = image.pixels
        for i in 0..<count {
            for c in 0..<3 {
                pixels[i * 4 + c] = UInt8(max(0, min(255, Float(image.pixels[i * 4 + c]) * gains[c] + 0.5)))
            }
        }
        return RGBAImage(width: image.width, height: image.height, pixels: pixels)
    }

    /// Contrast-limited adaptive histogram equalization on luma.
    public static func clahe(_ image: RGBAImage, tiles: Int = 8, clipLimit: Float = 2) -> RGBAImage {
        let (w, h) = (image.width, image.height)
        let luma = Luma.extract(image)
        let bins = 256
        var maps = [Float](repeating: 0, count: tiles * tiles * bins)

        for ty in 0..<tiles {
            for tx in 0..<tiles {
                let x0 = tx * w / tiles, x1 = (tx + 1) * w / tiles
                let y0 = ty * h / tiles, y1 = (ty + 1) * h / tiles
                var histogram = [Float](repeating: 0, count: bins)
                for y in y0..<y1 {
                    for x in x0..<x1 { histogram[min(bins - 1, Int(luma[y * w + x] * 255))] += 1 }
                }
                let pixelCount = Float((x1 - x0) * (y1 - y0))
                let limit = clipLimit * pixelCount / Float(bins)
                var excess: Float = 0
                for b in 0..<bins where histogram[b] > limit {
                    excess += histogram[b] - limit
                    histogram[b] = limit
                }
                var cumulative: Float = 0
                let base = (ty * tiles + tx) * bins
                for b in 0..<bins {
                    cumulative += histogram[b] + excess / Float(bins)
                    maps[base + b] = cumulative / pixelCount
                }
            }
        }

        var out = [Float](repeating: 0, count: w * h)
        for y in 0..<h {
            let fy = (Float(y) + 0.5) / Float(h) * Float(tiles) - 0.5
            let ty0 = max(0, Int(floor(fy))), ty1 = min(tiles - 1, ty0 + 1)
            let wy = max(0, min(1, fy - Float(ty0)))
            for x in 0..<w {
                let fx = (Float(x) + 0.5) / Float(w) * Float(tiles) - 0.5
                let tx0 = max(0, Int(floor(fx))), tx1 = min(tiles - 1, tx0 + 1)
                let wx = max(0, min(1, fx - Float(tx0)))
                let bin = min(bins - 1, Int(luma[y * w + x] * 255))
                let top = maps[(ty0 * tiles + tx0) * bins + bin] * (1 - wx) + maps[(ty0 * tiles + tx1) * bins + bin] * wx
                let bottom = maps[(ty1 * tiles + tx0) * bins + bin] * (1 - wx) + maps[(ty1 * tiles + tx1) * bins + bin] * wx
                out[y * w + x] = top * (1 - wy) + bottom * wy
            }
        }
        return Luma.apply(out, from: luma, to: image)
    }

    /// Edge-aware enhancement: a guided filter separates base from detail on luma,
    /// and detail is boosted only near real edges, so flat areas don't gain noise.
    public static func edgeAwareSharpen(_ image: RGBAImage, radius: Int = 2, epsilon: Float = 0.001, amount: Float = 0.8) -> RGBAImage {
        let (w, h) = (image.width, image.height)
        let luma = Luma.extract(image)
        let base = guidedFilter(luma, width: w, height: h, radius: radius, epsilon: epsilon)
        let edges = sobelMagnitude(luma, width: w, height: h)
        var out = [Float](repeating: 0, count: w * h)
        for i in 0..<(w * h) {
            let weight = min(1, edges[i] * 4)
            out[i] = max(0, min(1, base[i] + (luma[i] - base[i]) * (1 + amount * weight)))
        }
        return Luma.apply(out, from: luma, to: image)
    }

    /// Self-guided filter (He et al.), built on constant-time box means.
    static func guidedFilter(_ input: [Float], width w: Int, height h: Int, radius: Int, epsilon: Float) -> [Float] {
        let meanI = boxMean(input, width: w, height: h, radius: radius)
        let meanII = boxMean(input.map { $0 * $0 }, width: w, height: h, radius: radius)
        var a = [Float](repeating: 0, count: w * h)
        var b = [Float](repeating: 0, count: w * h)
        for i in 0..<(w * h) {
            let variance = meanII[i] - meanI[i] * meanI[i]
            a[i] = variance / (variance + epsilon)
            b[i] = (1 - a[i]) * meanI[i]
        }
        let meanA = boxMean(a, width: w, height: h, radius: radius)
        let meanB = boxMean(b, width: w, height: h, radius: radius)
        return (0..<(w * h)).map { meanA[$0] * input[$0] + meanB[$0] }
    }

    static func boxMean(_ input: [Float], width w: Int, height h: Int, radius r: Int) -> [Float] {
        var integral = [Double](repeating: 0, count: (w + 1) * (h + 1))
        for y in 0..<h {
            var row = 0.0
            for x in 0..<w {
                row += Double(input[y * w + x])
                integral[(y + 1) * (w + 1) + x + 1] = integral[y * (w + 1) + x + 1] + row
            }
        }
        var out = [Float](repeating: 0, count: w * h)
        for y in 0..<h {
            let y0 = max(0, y - r), y1 = min(h, y + r + 1)
            for x in 0..<w {
                let x0 = max(0, x - r), x1 = min(w, x + r + 1)
                let total = integral[y1 * (w + 1) + x1] - integral[y0 * (w + 1) + x1]
                    - integral[y1 * (w + 1) + x0] + integral[y0 * (w + 1) + x0]
                out[y * w + x] = Float(total / Double((x1 - x0) * (y1 - y0)))
            }
        }
        return out
    }

    static func sobelMagnitude(_ luma: [Float], width w: Int, height h: Int) -> [Float] {
        var out = [Float](repeating: 0, count: w * h)
        guard w > 2, h > 2 else { return out }
        for y in 1..<(h - 1) {
            for x in 1..<(w - 1) {
                let i = y * w + x
                let gx = luma[i - w + 1] + 2 * luma[i + 1] + luma[i + w + 1] - luma[i - w - 1] - 2 * luma[i - 1] - luma[i + w - 1]
                let gy = luma[i + w - 1] + 2 * luma[i + w] + luma[i + w + 1] - luma[i - w - 1] - 2 * luma[i - w] - luma[i - w + 1]
                out[i] = (gx * gx + gy * gy).squareRoot() / 8
            }
        }
        return out
    }

    /// High-quality resampling with vImage (Lanczos), used for export.
    public static func resized(_ image: RGBAImage, width: Int, height: Int) throws -> RGBAImage {
        var pixels = image.pixels
        var output = [UInt8](repeating: 255, count: width * height * 4)
        let error = pixels.withUnsafeMutableBytes { source in
            output.withUnsafeMutableBytes { destination in
                var src = vImage_Buffer(data: source.baseAddress, height: vImagePixelCount(image.height), width: vImagePixelCount(image.width), rowBytes: image.width * 4)
                var dst = vImage_Buffer(data: destination.baseAddress, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: width * 4)
                return vImageScale_ARGB8888(&src, &dst, nil, vImage_Flags(kvImageHighQualityResampling))
            }
        }
        guard error == kvImageNoError else { throw GeneratorError.imageEncodingFailed }
        return RGBAImage(width: width, height: height, pixels: output)
    }
}

/// One step of post-processing, kept so the app can show what changed and why.
public struct ClassicalStage: Sendable, Identifiable {
    public let name: String
    public let applied: Bool
    public let reason: String
    public let image: RGBAImage
    public let milliseconds: Double
    public var id: String { name }
}

public enum ClassicalPipeline {
    public struct Options: Sendable {
        public var colorCorrection = true
        public var contrast = true
        public var edges = true
        public init() {}
    }

    /// Each step is gated by a measurement, so it only runs when the image needs it.
    public static func process(_ image: RGBAImage, quality: QualityReport, prompt: ParsedPrompt, options: Options = Options()) -> [ClassicalStage] {
        var stages: [ClassicalStage] = []
        var current = image
        var watch = Stopwatch()

        if !options.colorCorrection {
            stages.append(ClassicalStage(name: "Color balance", applied: false, reason: "Turned off", image: current, milliseconds: 0))
        } else if prompt.namesColor {
            stages.append(ClassicalStage(name: "Color balance", applied: false, reason: "Skipped: the prompt names colors", image: current, milliseconds: 0))
        } else if quality.colorCast < 0.06 {
            stages.append(ClassicalStage(name: "Color balance", applied: false, reason: "Skipped: no color cast", image: current, milliseconds: 0))
        } else {
            _ = watch.lap()
            current = ImageOps.grayWorld(current)
            stages.append(ClassicalStage(name: "Color balance", applied: true, reason: "Gray-world for a color cast", image: current, milliseconds: watch.lap()))
        }

        if !options.contrast {
            stages.append(ClassicalStage(name: "Contrast", applied: false, reason: "Turned off", image: current, milliseconds: 0))
        } else if quality.isLowContrast {
            _ = watch.lap()
            current = ImageOps.clahe(current)
            stages.append(ClassicalStage(name: "Contrast", applied: true, reason: "CLAHE for low contrast", image: current, milliseconds: watch.lap()))
        } else {
            stages.append(ClassicalStage(name: "Contrast", applied: false, reason: "Skipped: contrast is fine", image: current, milliseconds: 0))
        }

        if options.edges {
            _ = watch.lap()
            current = ImageOps.edgeAwareSharpen(current)
            stages.append(ClassicalStage(name: "Edges", applied: true, reason: "Guided filter with edge-masked detail", image: current, milliseconds: watch.lap()))
        } else {
            stages.append(ClassicalStage(name: "Edges", applied: false, reason: "Turned off", image: current, milliseconds: 0))
        }
        return stages
    }
}
