import CoreML
import Foundation

public enum GeneratorError: Error, Equatable {
    case fileMissing(String)
    case notLoaded
    case emptyPrompt
    case badOutput(String)
    case invalidOutput(String)
    case imageEncodingFailed
}

/// Where the compiled models and tokenizer files live.
public struct ModelLocation: Sendable {
    public let models: URL
    public let tokenizer: URL

    public init(models: URL, tokenizer: URL) {
        self.models = models
        self.tokenizer = tokenizer
    }
}

public struct LoadReport: Sendable {
    public let textMs: Double
    public let unetMs: Double
    public let decoderMs: Double
    public var totalMs: Double { textMs + unetMs + decoderMs }
}

public struct AccelerationReport: Sendable {
    public let unetMs: Double
    public let reloads: Int
    public let isAccelerated: Bool
}

public struct GenerationTimings: Sendable {
    public var textMs: Double
    public var unetMs: Double
    public var decodeMs: Double
    public var totalMs: Double { textMs + unetMs + decodeMs }
}

public struct GeneratedCandidate: Sendable {
    public let seed: UInt64
    public let image: RGBAImage
    public let timings: GenerationTimings
}

public struct TokenizedPrompt: Sendable, Equatable {
    public let ids: [Int32]
    /// Tokens the prompt needed before being cut to fit 77.
    public let tokenCount: Int
    public var wasTruncated: Bool { tokenCount > Generator.maxTokens }
}

/// One-step text-to-image: text encoder -> UNet (once) -> TAESD decoder.
///
/// The sampler is the single denoising step at t=999, already baked into the
/// converted UNet, with no classifier-free guidance:
///     x0 = (z - sqrt(1 - alpha_bar) * eps) / sqrt(alpha_bar)
public actor Generator {
    public static let maxTokens = 77
    public static let latentChannels = 4
    public static let latentSize = 64
    public static let imageSize = 512
    static let sqrtOneMinusAlphaBar: Float = 0.99766725
    static let sqrtAlphaBar: Float = 0.06826489

    private let location: ModelLocation
    private let computeUnits: MLComputeUnits
    private let textComputeUnits: MLComputeUnits
    private var tokenizer: BPETokenizer?
    private var textEncoder: MLModel?
    private var unet: MLModel?
    private var decoder: MLModel?
    private var cached: (prompt: String, embedding: MLMultiArray)?

    public init(location: ModelLocation, computeUnits: MLComputeUnits = .cpuAndNeuralEngine, textComputeUnits: MLComputeUnits? = nil) {
        self.location = location
        self.computeUnits = computeUnits
        self.textComputeUnits = textComputeUnits ?? computeUnits
    }

    public var isLoaded: Bool { tokenizer != nil && textEncoder != nil && unet != nil && decoder != nil }

    /// Loads all models and runs each once, so the first real image doesn't pay
    /// for lazy Neural Engine setup. First launch includes the on-device compile.
    @discardableResult
    public func load(progress: @Sendable (String) -> Void = { _ in }) throws -> LoadReport {
        progress("Tokenizer")
        tokenizer = try BPETokenizer(
            mergesAt: fileURL(location.tokenizer, "merges.txt"),
            vocabularyAt: fileURL(location.tokenizer, "vocab.json")
        )

        progress("Text encoder")
        var watch = Stopwatch()
        textEncoder = try loadModel("TextEncoder.mlmodelc", units: textComputeUnits)
        let textMs = watch.lap()
        progress("Image model")
        unet = try loadModel("UNet.mlmodelc")
        let unetMs = watch.lap()
        progress("Decoder")
        decoder = try loadModel("TAESD.mlmodelc")
        let decoderMs = watch.lap()

        return LoadReport(textMs: textMs, unetMs: unetMs, decoderMs: decoderMs)
    }

    private func loadModel(_ name: String, units: MLComputeUnits? = nil) throws -> MLModel {
        let config = MLModelConfiguration()
        config.computeUnits = units ?? computeUnits
        return try MLModel(contentsOf: fileURL(location.models, name), configuration: config)
    }

    /// Core ML can silently fall back to the CPU if Neural Engine setup fails, which
    /// is about 13x slower with no error raised. This times the model, reloads it if
    /// it fell back, and reports whether it ended up accelerated.
    public func ensureAccelerated(thresholdMs: Double = 300, retries: Int = 2) throws -> AccelerationReport {
        var attempts = 0
        var unetMs = try probeUNetMs()
        while unetMs > thresholdMs, attempts < retries {
            attempts += 1
            unet = nil
            unet = try loadModel("UNet.mlmodelc")
            unetMs = try probeUNetMs()
        }
        return AccelerationReport(unetMs: unetMs, reloads: attempts, isAccelerated: unetMs <= thresholdMs)
    }

    /// The first call pays for lazy setup, so only the second is timed.
    private func probeUNetMs() throws -> Double {
        _ = try generate(prompt: "a photo", seed: 1)
        return try generate(prompt: "a photo", seed: 2).timings.unetMs
    }

    /// Releases model memory, e.g. on a system memory warning.
    public func unload() {
        textEncoder = nil
        unet = nil
        decoder = nil
        cached = nil
    }

    public func tokenize(_ prompt: String) throws -> TokenizedPrompt {
        guard let tokenizer else { throw GeneratorError.notLoaded }
        return Self.fit(tokenIDs: tokenizer.tokenize(input: prompt, minCount: Self.maxTokens).tokenIDs, endToken: tokenizer.tokenID(for: "<|endoftext|>") ?? 49407)
    }

    /// Pads are already applied by the tokenizer; over-long prompts keep their
    /// start and always end with the end-of-text token, as the encoder expects.
    static func fit(tokenIDs: [Int], endToken: Int) -> TokenizedPrompt {
        var kept = Array(tokenIDs.prefix(maxTokens))
        if tokenIDs.count > maxTokens { kept[maxTokens - 1] = endToken }
        return TokenizedPrompt(ids: kept.map(Int32.init), tokenCount: tokenIDs.count)
    }

    public func generate(prompt: String, seed: UInt64) throws -> GeneratedCandidate {
        guard let unet, let decoder else { throw GeneratorError.notLoaded }
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw GeneratorError.emptyPrompt }

        let (embedding, textMs) = try embedding(for: prompt)

        var noise = SeededNormal(seed: seed)
        let z = noise.fill(count: Self.latentChannels * Self.latentSize * Self.latentSize)
        let sample = try Self.latentArray(z)

        var watch = Stopwatch()
        let unetInput = try MLDictionaryFeatureProvider(dictionary: [
            "sample": MLFeatureValue(multiArray: sample),
            "encoder_hidden_states": MLFeatureValue(multiArray: embedding),
        ])
        guard let epsArray = try unet.prediction(from: unetInput).featureValue(for: "eps")?.multiArrayValue else {
            throw GeneratorError.badOutput("eps")
        }
        let eps = try Self.floats(from: epsArray)
        let unetMs = watch.lap()

        let x0 = zip(z, eps).map { ($0 - Self.sqrtOneMinusAlphaBar * $1) / Self.sqrtAlphaBar }

        let decoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "latents": MLFeatureValue(multiArray: try Self.latentArray(x0)),
        ])
        guard let imageArray = try decoder.prediction(from: decoderInput).featureValue(for: "image")?.multiArrayValue else {
            throw GeneratorError.badOutput("image")
        }
        let image = try Self.rgba(from: imageArray)
        let decodeMs = watch.lap()

        return GeneratedCandidate(
            seed: seed, image: image,
            timings: GenerationTimings(textMs: textMs, unetMs: unetMs, decodeMs: decodeMs)
        )
    }

    private func embedding(for prompt: String) throws -> (MLMultiArray, Double) {
        if let cached, cached.prompt == prompt { return (cached.embedding, 0) }
        guard let textEncoder else { throw GeneratorError.notLoaded }
        let tokens = try tokenize(prompt)
        let ids = try MLMultiArray(shape: [1, NSNumber(value: Self.maxTokens)], dataType: .int32)
        ids.withUnsafeMutableBufferPointer(ofType: Int32.self) { buffer, _ in
            for (i, id) in tokens.ids.enumerated() { buffer[i] = id }
        }
        var watch = Stopwatch()
        let input = try MLDictionaryFeatureProvider(dictionary: ["input_ids": MLFeatureValue(multiArray: ids)])
        guard let embedding = try textEncoder.prediction(from: input).featureValue(for: "emb")?.multiArrayValue else {
            throw GeneratorError.badOutput("emb")
        }
        let ms = watch.lap()
        cached = (prompt, embedding)
        return (embedding, ms)
    }

    private func fileURL(_ directory: URL, _ name: String) throws -> URL {
        let url = directory.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: url.path) else { throw GeneratorError.fileMissing(name) }
        return url
    }

    static func latentArray(_ values: [Float]) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, NSNumber(value: latentChannels), NSNumber(value: latentSize), NSNumber(value: latentSize)]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        array.withUnsafeMutableBufferPointer(ofType: Float32.self) { buffer, strides in
            // Inputs we allocate are contiguous, but honour strides anyway.
            precondition(strides.last == 1, "unexpected latent layout")
            for (i, value) in values.enumerated() { buffer[i] = value }
        }
        return array
    }

    /// Reads a [1, C, H, W] output respecting strides: the Neural Engine can pad rows.
    static func floats(from array: MLMultiArray) throws -> [Float] {
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        guard shape.count == 4 else { throw GeneratorError.badOutput("expected 4 dimensions, got \(shape)") }
        let (channels, height, width) = (shape[1], shape[2], shape[3])
        var out = [Float](repeating: 0, count: channels * height * width)

        func copy<T>(_ pointer: UnsafeRawPointer, as type: T.Type, convert: (T) -> Float) {
            let source = pointer.assumingMemoryBound(to: T.self)
            var index = 0
            for c in 0..<channels {
                for y in 0..<height {
                    let row = c * strides[1] + y * strides[2]
                    for x in 0..<width {
                        out[index] = convert(source[row + x * strides[3]])
                        index += 1
                    }
                }
            }
        }

        try array.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { throw GeneratorError.badOutput("empty") }
            switch array.dataType {
            case .float32: copy(base, as: Float32.self) { $0 }
            case .float16: copy(base, as: Float16.self) { Float($0) }
            case .double: copy(base, as: Double.self) { Float($0) }
            default: throw GeneratorError.badOutput("unsupported data type \(array.dataType.rawValue)")
            }
        }
        return out
    }

    /// TAESD output [-1, 1] in planar RGB to interleaved 8-bit RGBA.
    /// Non-finite values mean the model failed, so they are errors rather than black pixels.
    static func rgba(from array: MLMultiArray) throws -> RGBAImage {
        let shape = array.shape.map(\.intValue)
        guard shape.count == 4, shape[1] == 3 else { throw GeneratorError.badOutput("image shape \(shape)") }
        let (height, width) = (shape[2], shape[3])
        let planar = try floats(from: array)
        var pixels = [UInt8](repeating: 255, count: width * height * 4)
        for i in 0..<(width * height) {
            for c in 0..<3 {
                let value = planar[c * width * height + i]
                guard value.isFinite else { throw GeneratorError.invalidOutput("non-finite pixel") }
                pixels[i * 4 + c] = UInt8(max(0, min(1, (value + 1) / 2)) * 255 + 0.5)
            }
        }
        return RGBAImage(width: width, height: height, pixels: pixels)
    }
}

struct Stopwatch {
    private var last = ContinuousClock.now

    /// Milliseconds since the previous lap, or since creation.
    mutating func lap() -> Double {
        let now = ContinuousClock.now
        let elapsed = last.duration(to: now).components
        last = now
        return Double(elapsed.seconds) * 1000 + Double(elapsed.attoseconds) / 1e15
    }
}
