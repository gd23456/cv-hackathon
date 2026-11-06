import CoreML
import Foundation
import Vision

public struct VerifierLocation: Sendable {
    public let directory: URL
    public let tokenizer: URL

    public init(directory: URL, tokenizer: URL) {
        self.directory = directory
        self.tokenizer = tokenizer
    }
}

public struct Verification: Sendable {
    public let checks: [CheckResult]
    public let detections: [Detection]
    /// Cosine similarity between the prompt and image, from MobileCLIP.
    public let alignment: Float
    public let detectMs: Double
    public let alignMs: Double
}

/// On-device checking: Apple's YOLOv3-Tiny, MobileCLIP S0 and an NSFW classifier.
public actor Verifier {
    private let location: VerifierLocation
    private let computeUnits: MLComputeUnits
    private var detector: VNCoreMLModel?
    private var safety: VNCoreMLModel?
    private var imageEncoder: MLModel?
    private var textEncoder: MLModel?
    private var tokenizer: BPETokenizer?
    private var cachedText: (prompt: String, embedding: [Float])?

    public init(location: VerifierLocation, computeUnits: MLComputeUnits = .cpuAndNeuralEngine) {
        self.location = location
        self.computeUnits = computeUnits
    }

    public var isLoaded: Bool { detector != nil && imageEncoder != nil && textEncoder != nil }

    public func load() throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        func url(_ name: String) throws -> URL {
            let url = location.directory.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else { throw GeneratorError.fileMissing(name) }
            return url
        }
        let yolo = try VNCoreMLModel(for: MLModel(contentsOf: url("YOLOv3TinyInt8LUT.mlmodelc"), configuration: config))
        yolo.featureProvider = try MLDictionaryFeatureProvider(dictionary: ["iouThreshold": 0.45, "confidenceThreshold": 0.2])
        detector = yolo
        safety = try VNCoreMLModel(for: MLModel(contentsOf: url("NSFW.mlmodelc"), configuration: config))
        imageEncoder = try MLModel(contentsOf: url("mobileclip_s0_image.mlmodelc"), configuration: config)
        textEncoder = try MLModel(contentsOf: url("mobileclip_s0_text.mlmodelc"), configuration: config)
        // OpenCLIP pads with id 0, which is the "!" token, unlike Stable Diffusion's encoder.
        tokenizer = try BPETokenizer(
            mergesAt: location.tokenizer.appendingPathComponent("merges.txt"),
            vocabularyAt: location.tokenizer.appendingPathComponent("vocab.json"),
            padToken: "!"
        )
    }

    public func unload() {
        detector = nil
        safety = nil
        imageEncoder = nil
        textEncoder = nil
        cachedText = nil
    }

    public func verify(_ image: RGBAImage, prompt: String, parsed: ParsedPrompt) throws -> Verification {
        var watch = Stopwatch()
        let detections = try detect(in: image)
        let detectMs = watch.lap()
        let alignment = try align(image, prompt: prompt)
        let alignMs = watch.lap()
        return Verification(
            checks: Checklist.evaluate(parsed, detections: detections, image: image),
            detections: detections, alignment: alignment, detectMs: detectMs, alignMs: alignMs
        )
    }

    /// Probability that the image is unsafe, or nil if the classifier isn't loaded.
    public func unsafeProbability(of image: RGBAImage) throws -> Float? {
        guard let safety, let cgImage = image.cgImage else { return nil }
        let request = VNCoreMLRequest(model: safety)
        request.imageCropAndScaleOption = .centerCrop
        try VNImageRequestHandler(cgImage: cgImage, orientation: .up).perform([request])
        let results = request.results as? [VNClassificationObservation] ?? []
        return results.first { $0.identifier == "NSFW" }?.confidence
    }

    private func detect(in image: RGBAImage) throws -> [Detection] {
        guard let detector, let cgImage = image.cgImage else { throw GeneratorError.notLoaded }
        let request = VNCoreMLRequest(model: detector)
        request.imageCropAndScaleOption = .scaleFill
        try VNImageRequestHandler(cgImage: cgImage, orientation: .up).perform([request])
        let observations = request.results as? [VNRecognizedObjectObservation] ?? []
        return observations.compactMap { observation in
            guard let top = observation.labels.first else { return nil }
            let box = observation.boundingBox
            return Detection(
                label: top.identifier.replacingOccurrences(of: "diningtable", with: "dining table"),
                confidence: top.confidence,
                box: CGRect(x: box.minX, y: 1 - box.maxY, width: box.width, height: box.height)
            )
        }
    }

    private func align(_ image: RGBAImage, prompt: String) throws -> Float {
        guard let imageEncoder, let cgImage = image.cgImage else { throw GeneratorError.notLoaded }
        let text = try textEmbedding(prompt)
        let value = try MLFeatureValue(
            cgImage: cgImage, pixelsWide: 256, pixelsHigh: 256,
            pixelFormatType: kCVPixelFormatType_32ARGB, options: nil
        )
        let input = try MLDictionaryFeatureProvider(dictionary: ["image": value])
        guard let array = try imageEncoder.prediction(from: input).featureValue(for: "final_emb_1")?.multiArrayValue else {
            throw GeneratorError.badOutput("image embedding")
        }
        return Self.cosine(Self.vector(from: array), text)
    }

    private func textEmbedding(_ prompt: String) throws -> [Float] {
        if let cachedText, cachedText.prompt == prompt { return cachedText.embedding }
        guard let textEncoder, let tokenizer else { throw GeneratorError.notLoaded }
        let fitted = Generator.fit(
            tokenIDs: tokenizer.tokenize(input: prompt, minCount: Generator.maxTokens).tokenIDs,
            endToken: tokenizer.tokenID(for: "<|endoftext|>") ?? 49407
        )
        let ids = try MLMultiArray(shape: [1, 77], dataType: .int32)
        ids.withUnsafeMutableBufferPointer(ofType: Int32.self) { buffer, _ in
            for (i, id) in fitted.ids.enumerated() { buffer[i] = id }
        }
        let input = try MLDictionaryFeatureProvider(dictionary: ["text": MLFeatureValue(multiArray: ids)])
        guard let array = try textEncoder.prediction(from: input).featureValue(for: "final_emb_1")?.multiArrayValue else {
            throw GeneratorError.badOutput("text embedding")
        }
        let embedding = Self.vector(from: array)
        cachedText = (prompt, embedding)
        return embedding
    }

    static func vector(from array: MLMultiArray) -> [Float] {
        (0..<array.count).map { array[$0].floatValue }
    }

    static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0, normA: Float = 0, normB: Float = 0
        for i in 0..<min(a.count, b.count) {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        guard normA > 0, normB > 0 else { return 0 }
        return dot / (normA.squareRoot() * normB.squareRoot())
    }
}
