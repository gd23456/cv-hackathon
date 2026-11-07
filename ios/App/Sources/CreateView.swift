import KshanaCore
import SwiftUI

struct CreateView: View {
    @Bindable var studio: Studio
    @FocusState private var promptFocused: Bool
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    @State private var network = NetworkStatus()
    @State private var voice = VoiceInput()

    // The same tested prompts as the website.
    private let examples = [
        "a red apple to the left of a blue cup",
        "a lighthouse on a cliff at sunset",
        "a golden retriever puppy on green grass",
        "a bowl of ramen on a dark table",
    ]

    private var motion: Bool { !reduceMotion && studio.effectsAllowed }
    private var working: Bool { studio.phase == .generating }

    var body: some View {
        NavigationStack {
            ZStack {
                GraphiteBackground()
                if motion, working {
                    WarpField(cards: studio.flyingCards)
                        .offset(y: -130)
                }
                ScrollView {
                    VStack(alignment: .leading, spacing: 14) {
                        header
                        canvas
                        if let notice = studio.notice {
                            Label(notice, systemImage: "info.circle")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }
                        if studio.plan != nil { RunReportView(studio: studio) }
                    }
                    .padding(.horizontal)
                    .padding(.top, 6)
                    .padding(.bottom, 12)
                }
                .scrollDismissesKeyboard(.interactively)
            }
            .safeAreaInset(edge: .bottom) { composer }
            .toolbar(.hidden, for: .navigationBar)
            .sensoryFeedback(.success, trigger: studio.completedRuns)
        }
    }

    private var header: some View {
        HStack {
            Text("Kshana").font(.system(size: 28, weight: .bold)).tracking(-0.5)
            Spacer()
            HStack(spacing: 5) {
                Image(systemName: network.isOffline ? "airplane" : "cpu")
                Text(network.isOffline ? "No network" : (studio.acceleration?.isAccelerated == false ? "On CPU" : "On device"))
            }
            .font(.caption.weight(.medium))
            .foregroundStyle(network.isOffline ? Color(red: 0.35, green: 0.78, blue: 0.5) : .secondary)
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(Color.kshanaPanel, in: .capsule)
            .animation(.snappy, value: network.isOffline)
            if let image = studio.exportImage {
                ShareLink(item: Image(uiImage: image), preview: SharePreview("AI-generated image", image: Image(uiImage: image))) {
                    Image(systemName: "square.and.arrow.up").padding(.leading, 8)
                }
            }
        }
    }

    private var canvas: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 16).fill(Color.kshanaPanel)
            if working, studio.image == nil {
                GeneratingView(prompt: studio.prompt, animated: motion)
                    .transition(.opacity)
            }
            if let image = studio.image {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .modifier(ImageArrival(revision: studio.imageRevision, enabled: motion))
                    .clipShape(.rect(cornerRadius: 16))
                    .contentShape(.rect)
                    .gesture(exploreGesture)
                    .overlay {
                        if working, motion {
                            ScanLine().clipShape(.rect(cornerRadius: 16))
                        }
                    }
                    .overlay(alignment: .bottomLeading) {
                        Text("AI generated")
                            .font(.caption2.weight(.medium))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(.black.opacity(0.55), in: .capsule)
                            .foregroundStyle(.white)
                            .padding(10)
                    }
                    .accessibilityLabel("Generated image: \(studio.prompt)")
            }
            canvasStatus
        }
        .aspectRatio(1, contentMode: .fit)
        .overlay(RoundedRectangle(cornerRadius: 16).strokeBorder(.white.opacity(0.08)))
        .sensoryFeedback(.impact(weight: .light), trigger: studio.imageRevision)
    }

    @ViewBuilder private var canvasStatus: some View {
        switch studio.phase {
        case .preparing(let stage):
            VStack(spacing: 10) {
                ProgressView()
                Text("Preparing \(stage)").font(.subheadline.weight(.medium))
                Text("The first launch sets up the Neural Engine and can take a minute. After that it's much quicker.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            .padding(24)
        case .failed(let message):
            Label(message, systemImage: "exclamationmark.triangle")
                .font(.subheadline)
                .multilineTextAlignment(.center)
                .padding(24)
        case .generating where studio.image == nil:
            EmptyView()
        case .ready where studio.image == nil:
            Text("Your image appears here")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        default:
            EmptyView()
        }
    }

    /// Dragging across the image travels through the model's starting noise.
    private var exploreGesture: some Gesture {
        DragGesture(minimumDistance: 14)
            .onChanged { drag in
                guard abs(drag.translation.width) > abs(drag.translation.height) || studio.isExploring else { return }
                studio.explore(position: drag.translation.width / 160)
            }
            .onEnded { _ in studio.endExploring() }
    }

    private var statusLine: String? {
        if let message = voice.message { return message }
        if voice.isListening { return "Listening. Tap the microphone to stop." }
        if let notice = studio.quickNotice { return notice }
        if studio.liveMode, let ms = studio.quickMs {
            return String(format: "Live: %.0f ms per image, safety check on. Press send for the full search.", ms)
        }
        if studio.isExploring { return "Exploring. Press send to search and check." }
        if studio.image != nil, !working, !studio.liveMode { return "Drag the image sideways to explore variations." }
        return nil
    }

    private var composer: some View {
        VStack(spacing: 10) {
            if let line = statusLine {
                Text(line)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal)
            }
            if studio.prompt.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(examples, id: \.self) { example in
                            Button(example) { studio.prompt = example }
                                .font(.caption)
                                .padding(.horizontal, 11)
                                .padding(.vertical, 7)
                                .background(Color.kshanaPanel, in: .capsule)
                                .foregroundStyle(.primary)
                        }
                    }
                    .padding(.horizontal)
                }
            }
            HStack(spacing: 10) {
                Button {
                    studio.liveMode.toggle()
                } label: {
                    Label("Live", systemImage: "bolt.fill")
                        .font(.footnote.weight(.semibold))
                        .padding(.horizontal, 12)
                        .frame(height: 32)
                        .background(studio.liveMode ? Color.kshanaAccent : Color.kshanaPanel, in: .capsule)
                        .foregroundStyle(studio.liveMode ? .white : .primary)
                }
                .accessibilityValue(studio.liveMode ? "On" : "Off")
                .sensoryFeedback(.selection, trigger: studio.liveMode)
                Picker("Effort", selection: $studio.budget) {
                    ForEach(Budget.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
            }
            .padding(.horizontal)
            HStack(alignment: .bottom, spacing: 10) {
                TextField("Describe an image", text: $studio.prompt, axis: .vertical)
                    .lineLimit(1...3)
                    .focused($promptFocused)
                    .submitLabel(.go)
                    .onSubmit(generate)
                    .onChange(of: studio.prompt) { studio.promptDidChange() }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 11)
                    .background(Color.kshanaPanel, in: .rect(cornerRadius: 14))
                    .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(.white.opacity(0.1)))
                Button {
                    Task { await voice.toggle { studio.prompt = $0 } }
                } label: {
                    Image(systemName: voice.isListening ? "waveform" : "mic")
                        .symbolEffect(.variableColor.iterative, isActive: voice.isListening && !reduceMotion)
                        .frame(width: 44, height: 44)
                        .background(voice.isListening ? Color.kshanaAccent : Color.kshanaPanel, in: .circle)
                        .foregroundStyle(voice.isListening ? .white : .primary)
                }
                .accessibilityLabel(voice.isListening ? "Stop listening" : "Speak a prompt")
                Button(action: generate) {
                    Group {
                        if working {
                            ProgressView().tint(.white)
                        } else {
                            Image(systemName: "arrow.up").font(.body.weight(.bold))
                        }
                    }
                    .frame(width: 44, height: 44)
                    .background(studio.canGenerate || working ? Color.kshanaAccent : Color.kshanaPanel, in: .circle)
                    .foregroundStyle(.white)
                }
                .disabled(!studio.canGenerate)
                .accessibilityLabel(working ? "Generating" : "Generate")
            }
            .padding(.horizontal)
        }
        .padding(.top, 10)
        .padding(.bottom, 8)
        .background(.bar)
    }

    private func generate() {
        promptFocused = false
        if voice.isListening { voice.stop() }
        studio.generate()
    }
}
