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

/// An estimate of how believable the hands in an image are, using Apple's on-device pose models.
/// It catches many broken hands, not all of them.
public struct HandReport: Sendable, Equatable {
    public let plausibleHands: Int
    public let brokenHands: Int
    /// Visible wrists with no recognisable hand on the end.
    public let missingHands: Int

    public init(plausibleHands: Int, brokenHands: Int, missingHands: Int) {
        self.plausibleHands = plausibleHands
        self.brokenHands = brokenHands
        self.missingHands = missingHands
    }

    public var penalty: Double { min(0.35, Double(brokenHands) * 0.12 + Double(missingHands) * 0.08) }

    public var check: CheckResult? {
        if brokenHands + missingHands > 0 { return CheckResult(title: "hands look natural", state: .failed) }
        return plausibleHands > 0 ? CheckResult(title: "hands look natural", state: .passed) : nil
    }
}

public struct Verification: Sendable {
    public let checks: [CheckResult]
    public let detections: [Detection]
    /// Cosine similarity between the prompt and image, from MobileCLIP.
    public let alignment: Float
    public let hands: HandReport?
    public let detectMs: Double
    public let alignMs: Double
}

/// On-device checking of prompt facts: Apple's YOLOv3-Tiny and MobileCLIP S0.
public actor Verifier {
    private let location: VerifierLocation
    private let computeUnits: MLComputeUnits
    private var detector: VNCoreMLModel?
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
        let hasPerson = detections.contains { $0.label == "person" && $0.confidence >= Checklist.presenceConfidence }
        let hands = hasPerson ? handReport(for: image) : nil
        let handMs = watch.lap()
        return Verification(
            checks: Checklist.evaluate(parsed, detections: detections, image: image),
            detections: detections, alignment: alignment, hands: hands, detectMs: detectMs + handMs, alignMs: alignMs
        )
    }

    private func handReport(for image: RGBAImage) -> HandReport? {
        guard let cgImage = image.cgImage else { return nil }
        let handRequest = VNDetectHumanHandPoseRequest()
        handRequest.maximumHandCount = 6
        let bodyRequest = VNDetectHumanBodyPoseRequest()
        guard (try? VNImageRequestHandler(cgImage: cgImage, orientation: .up).perform([handRequest, bodyRequest])) != nil else { return nil }

        var plausible = 0, broken = 0
        var handWrists: [CGPoint] = []
        for hand in handRequest.results ?? [] {
            guard let points = try? hand.recognizedPoints(.all) else { continue }
            if let wrist = points[.wrist], wrist.confidence >= 0.3 { handWrists.append(wrist.location) }
            if Self.handLooksNatural(points) { plausible += 1 } else { broken += 1 }
        }

        var missing = 0
        for body in bodyRequest.results ?? [] {
            for joint in [VNHumanBodyPoseObservation.JointName.leftWrist, .rightWrist] {
                guard let wrist = try? body.recognizedPoint(joint), wrist.confidence >= 0.5 else { continue }
                let matched = handWrists.contains { hypot($0.x - wrist.location.x, $0.y - wrist.location.y) < 0.12 }
                if !matched { missing += 1 }
            }
        }
        return HandReport(plausibleHands: plausible, brokenHands: broken, missingHands: missing)
    }

    /// A believable hand shows nearly all 21 joints, with fingers in proportion to the palm.
    static func handLooksNatural(_ points: [VNHumanHandPoseObservation.JointName: VNRecognizedPoint]) -> Bool {
        let visible = points.values.filter { $0.confidence >= 0.3 }
        guard visible.count >= 18 else { return false }
        let mean = visible.reduce(0) { $0 + $1.confidence } / Float(visible.count)
        guard mean >= 0.45, let wrist = points[.wrist], let palm = points[.middleMCP] else { return false }
        let palmLength = hypot(wrist.location.x - palm.location.x, wrist.location.y - palm.location.y)
        guard palmLength > 0.005 else { return false }
        let fingers: [(VNHumanHandPoseObservation.JointName, VNHumanHandPoseObservation.JointName)] = [
            (.indexMCP, .indexTip), (.middleMCP, .middleTip), (.ringMCP, .ringTip), (.littleMCP, .littleTip),
        ]
        for (base, tip) in fingers {
            guard let b = points[base], let t = points[tip] else { continue }
            let ratio = hypot(b.location.x - t.location.x, b.location.y - t.location.y) / palmLength
            if ratio > 1.8 { return false }
        }
        return true
    }

    /// Apple's on-device aesthetics score, from -1 to 1, used to choose between close finalists.
    public func aesthetics(of image: RGBAImage) -> Float? {
        guard let cgImage = image.cgImage else { return nil }
        let request = VNCalculateImageAestheticsScoresRequest()
        guard (try? VNImageRequestHandler(cgImage: cgImage, orientation: .up).perform([request])) != nil else { return nil }
        return request.results?.first?.overallScore
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
