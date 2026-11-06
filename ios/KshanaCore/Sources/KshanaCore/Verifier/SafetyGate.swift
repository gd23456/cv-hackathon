import CoreML
import Foundation

/// Every image passes here before it can reach the screen, using
/// Marqo/nsfw-image-detection-384. It fails closed: if the model is missing
/// or anything goes wrong, the image counts as unsafe.
public actor SafetyGate {
    /// From eval/redteam.py, classifier only, so stricter than the server, which adds a CLIP check.
    /// Skin-heavy images get the stricter limit.
    static let threshold: Float = 0.15
    static let skinThreshold: Float = 0.1
    static let skinShare = 0.35

    private let modelURL: URL?
    private var model: MLModel?

    public init(modelURL: URL?) {
        self.modelURL = modelURL
    }

    public var isLoaded: Bool { model != nil }

    public func load() throws {
        guard let modelURL else { throw GeneratorError.fileMissing("nsfw_marqo.mlmodelc") }
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        model = try MLModel(contentsOf: modelURL, configuration: config)
        // Warm up, so the first real image isn't slowed by lazy setup.
        _ = try probability(of: RGBAImage(width: 8, height: 8, pixels: [UInt8](repeating: 128, count: 8 * 8 * 4)))
    }

    public func allows(_ image: RGBAImage) -> Bool {
        guard let probability = try? probability(of: image) else { return false }
        let limit = Self.skinRatio(image) >= Self.skinShare ? Self.skinThreshold : Self.threshold
        return probability < limit
    }

    private func probability(of image: RGBAImage) throws -> Float {
        guard let model else { throw GeneratorError.notLoaded }
        let size = 384
        let small = try ImageOps.resized(image, width: size, height: size)
        let input = try MLMultiArray(shape: [1, 3, NSNumber(value: size), NSNumber(value: size)], dataType: .float32)
        input.withUnsafeMutableBufferPointer(ofType: Float32.self) { buffer, _ in
            for c in 0..<3 {
                for i in 0..<(size * size) {
                    buffer[c * size * size + i] = Float32(small.pixels[i * 4 + c])
                }
            }
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(multiArray: input)])
        guard let probs = try model.prediction(from: provider).featureValue(for: "probs")?.multiArrayValue, probs.count == 2 else {
            throw GeneratorError.badOutput("probs")
        }
        let unsafe = probs[0].floatValue
        guard unsafe.isFinite else { throw GeneratorError.invalidOutput("safety score") }
        return unsafe
    }

    /// Classical signal: share of pixels in the usual YCrCb skin-tone range.
    static func skinRatio(_ image: RGBAImage) -> Double {
        let count = image.width * image.height
        guard count > 0 else { return 0 }
        var skin = 0
        for i in 0..<count {
            let r = Float(image.pixels[i * 4]), g = Float(image.pixels[i * 4 + 1]), b = Float(image.pixels[i * 4 + 2])
            let y = 0.299 * r + 0.587 * g + 0.114 * b
            let cr = (r - y) * 0.713 + 128
            let cb = (b - y) * 0.564 + 128
            if y >= 40, cr >= 135, cr <= 180, cb >= 85, cb <= 135 { skin += 1 }
        }
        return Double(skin) / Double(count)
    }
}
