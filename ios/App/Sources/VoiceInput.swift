import AVFoundation
import Observation
@preconcurrency import Speech

/// Dictates a prompt using Apple's on-device speech recognition, with no network.
@Observable
final class VoiceInput {
    private(set) var isListening = false
    private(set) var message: String?

    private let engine = AVAudioEngine()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?

    func toggle(onText: @escaping @MainActor (String) -> Void) async {
        if isListening {
            stop()
            return
        }
        message = nil
        guard await Self.requestSpeechPermission() else {
            message = "Allow speech recognition in Settings to use voice."
            return
        }
        guard await AVAudioApplication.requestRecordPermission() else {
            message = "Allow the microphone in Settings to use voice."
            return
        }
        guard let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US")), recognizer.isAvailable, recognizer.supportsOnDeviceRecognition else {
            message = "On-device speech isn't available on this iPhone. Check Keyboard Dictation languages in Settings."
            return
        }
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.record, mode: .measurement, options: .duckOthers)
            try session.setActive(true, options: .notifyOthersOnDeactivation)
            let request = SFSpeechAudioBufferRecognitionRequest()
            request.requiresOnDeviceRecognition = true
            request.shouldReportPartialResults = true
            self.request = request
            task = Self.startRecognition(recognizer, request: request) { [weak self] text, finished in
                Task { @MainActor in
                    if let text { onText(text) }
                    if finished { self?.stop() }
                }
            }
            Self.installTap(on: engine, request: request)
            engine.prepare()
            try engine.start()
            isListening = true
        } catch {
            message = "Voice couldn't start. Try typing instead."
            stop()
        }
    }

    func stop() {
        if engine.isRunning { engine.stop() }
        engine.inputNode.removeTap(onBus: 0)
        request?.endAudio()
        task?.cancel()
        request = nil
        task = nil
        isListening = false
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    // These run on audio and speech threads, so they must not be tied to the main actor.
    private nonisolated static func requestSpeechPermission() async -> Bool {
        await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { continuation.resume(returning: $0 == .authorized) }
        }
    }

    private nonisolated static func installTap(on engine: AVAudioEngine, request: SFSpeechAudioBufferRecognitionRequest) {
        let input = engine.inputNode
        nonisolated(unsafe) let target = request
        input.installTap(onBus: 0, bufferSize: 1024, format: input.outputFormat(forBus: 0)) { buffer, _ in
            target.append(buffer)
        }
    }

    private nonisolated static func startRecognition(
        _ recognizer: SFSpeechRecognizer, request: SFSpeechAudioBufferRecognitionRequest,
        handler: @escaping @Sendable (String?, Bool) -> Void
    ) -> SFSpeechRecognitionTask {
        recognizer.recognitionTask(with: request) { result, error in
            handler(result?.bestTranscription.formattedString, error != nil || result?.isFinal == true)
        }
    }
}
