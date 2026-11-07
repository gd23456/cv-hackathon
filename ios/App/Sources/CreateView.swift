import KshanaCore
import SwiftUI

struct CreateView: View {
    @Bindable var studio: Studio
    @FocusState private var promptFocused: Bool
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private let examples = [
        "a red apple to the left of a blue cup",
        "two dogs sitting on a couch",
        "a bicycle under a tree in the rain",
        "a cat above a yellow book",
    ]

    private var motion: Bool { !reduceMotion && studio.effectsAllowed }
    private var working: Bool { studio.phase == .generating }

    var body: some View {
        NavigationStack {
            ZStack {
                AuroraBackground(energy: working ? 1 : 0, animated: motion)
                if motion {
                    WarpField(cards: studio.flyingCards)
                        .offset(y: -150)
                }
                ScrollView {
                    VStack(alignment: .leading, spacing: 18) {
                        header
                        canvas
                        if let notice = studio.notice {
                            Label(notice, systemImage: "info.circle")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }
                        if !studio.checks.isEmpty { checklist }
                        composer
                        if studio.plan != nil || studio.summary != nil { runDetails }
                    }
                    .padding()
                }
                .scrollDismissesKeyboard(.interactively)
                if !reduceMotion {
                    InkBurst(trigger: studio.burstCount)
                }
            }
            .toolbar(.hidden, for: .navigationBar)
            .sensoryFeedback(.success, trigger: studio.completedRuns)
        }
    }

    private var header: some View {
        HStack(alignment: .center) {
            InkTitle(text: "Kshana", active: working && motion)
            Spacer()
            Label(studio.acceleration?.isAccelerated == false ? "CPU" : "On this iPhone", systemImage: "iphone.gen3")
                .font(.caption2.weight(.semibold))
                .padding(.horizontal, 9)
                .padding(.vertical, 6)
                .background(.ultraThinMaterial, in: .capsule)
            if let image = studio.exportImage {
                ShareLink(item: Image(uiImage: image), preview: SharePreview("AI-generated image", image: Image(uiImage: image))) {
                    Image(systemName: "square.and.arrow.up")
                        .padding(8)
                        .background(.ultraThinMaterial, in: .circle)
                }
            }
        }
    }

    private var canvas: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 20).fill(.ultraThinMaterial)
            if working, studio.image == nil, motion {
                NoiseField()
                    .clipShape(.rect(cornerRadius: 20))
                    .transition(.opacity)
            }
            if let image = studio.image {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .modifier(ImageArrival(revision: studio.imageRevision, enabled: motion))
                    .clipShape(.rect(cornerRadius: 20))
                    .overlay {
                        if !studio.detections.isEmpty {
                            DetectionOverlay(detections: studio.detections, revision: studio.imageRevision)
                                .clipShape(.rect(cornerRadius: 20))
                        }
                    }
                    .overlay {
                        if working, motion {
                            ScanLine().clipShape(.rect(cornerRadius: 20))
                        }
                    }
                    .overlay(alignment: .bottomLeading) {
                        Text("AI generated")
                            .font(.caption2.weight(.medium))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(.ultraThinMaterial, in: .capsule)
                            .padding(10)
                    }
                    .accessibilityLabel("Generated image: \(studio.prompt)")
            }
            canvasStatus
        }
        .aspectRatio(1, contentMode: .fit)
        .shadow(color: Color(red: 0.93, green: 0, blue: 0.55).opacity(working ? 0.55 : 0.2), radius: working ? 34 : 18)
        .animation(reduceMotion ? nil : .easeInOut(duration: 0.5), value: working)
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
            if !motion { ProgressView() }
        case .ready where studio.image == nil:
            Text("Your image appears here")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        default:
            EmptyView()
        }
    }

    private var checklist: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Prompt check").font(.footnote.weight(.semibold)).foregroundStyle(.secondary)
            FlowLayout(spacing: 6) {
                ForEach(studio.checks) { check in
                    CheckChip(check: check)
                        .transition(.scale(scale: 0.85).combined(with: .opacity))
                }
            }
            .animation(reduceMotion ? nil : .spring(duration: 0.28, bounce: 0.2), value: studio.checks)
        }
    }

    private var composer: some View {
        VStack(alignment: .leading, spacing: 12) {
            TextField("Describe an image", text: $studio.prompt, axis: .vertical)
                .lineLimit(1...4)
                .focused($promptFocused)
                .submitLabel(.go)
                .onSubmit(generate)
                .padding(14)
                .background(.ultraThinMaterial, in: .rect(cornerRadius: 16))
            if studio.prompt.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(examples, id: \.self) { example in
                            Button(example) { studio.prompt = example }
                                .buttonStyle(.bordered)
                                .font(.caption)
                        }
                    }
                }
            }
            Picker("Effort", selection: $studio.budget) {
                ForEach(Budget.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            Button(action: generate) {
                HStack(spacing: 8) {
                    Image(systemName: "sparkles")
                        .symbolEffect(.variableColor.iterative, isActive: working && !reduceMotion)
                    if working {
                        Text(studio.triedCount > 0 ? "Checked \(studio.triedCount) images" : "Generating")
                            .contentTransition(.numericText(value: Double(studio.triedCount)))
                            .animation(.snappy(duration: 0.15), value: studio.triedCount)
                    } else {
                        Text("Generate")
                    }
                }
            }
            .buttonStyle(InkButtonStyle(working: working && !reduceMotion))
            .opacity(studio.canGenerate || working ? 1 : 0.5)
            .disabled(!studio.canGenerate)
        }
    }

    private var runDetails: some View {
        VStack(alignment: .leading, spacing: 4) {
            if let plan = studio.plan { Text(plan.reason) }
            if let summary = studio.summary {
                Text(verbatim: String(format: "%d images in %.1f s. First image in %.0f ms. No network used.", summary.candidatesTried, summary.elapsedMs / 1000, summary.firstImageMs))
                    .monospacedDigit()
            }
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }

    private func generate() {
        promptFocused = false
        studio.generate()
    }
}

private struct CheckChip: View {
    let check: CheckResult

    var body: some View {
        Label(check.title, systemImage: symbol)
            .font(.caption)
            .padding(.horizontal, 9)
            .padding(.vertical, 5)
            .background(color.opacity(0.14), in: .capsule)
            .foregroundStyle(color)
            .accessibilityLabel("\(check.title): \(spoken)")
    }

    private var symbol: String {
        switch check.state {
        case .passed: "checkmark"
        case .failed: "xmark"
        case .notVerifiable: "questionmark"
        }
    }

    private var color: Color {
        switch check.state {
        case .passed: .green
        case .failed: .red
        case .notVerifiable: .secondary
        }
    }

    private var spoken: String {
        switch check.state {
        case .passed: "found"
        case .failed: "not found"
        case .notVerifiable: "can't be checked"
        }
    }
}

/// Moving grain while the first image is on its way, like the noise the model starts from.
private struct NoiseField: View {
    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 15)) { context in
            Canvas { canvas, size in
                let cells = 32
                let cell = CGSize(width: size.width / CGFloat(cells), height: size.height / CGFloat(cells))
                var noise = SeededNormal(seed: UInt64(context.date.timeIntervalSinceReferenceDate * 15))
                let values = noise.fill(count: cells * cells)
                for y in 0..<cells {
                    for x in 0..<cells {
                        let shade = max(0, min(1, 0.5 + Double(values[y * cells + x]) * 0.22))
                        canvas.fill(
                            Path(CGRect(x: CGFloat(x) * cell.width, y: CGFloat(y) * cell.height, width: cell.width + 0.5, height: cell.height + 0.5)),
                            with: .color(white: shade)
                        )
                    }
                }
            }
        }
        .accessibilityHidden(true)
    }
}

/// Wraps chips onto as many lines as they need.
struct FlowLayout: Layout {
    var spacing: CGFloat = 6

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? .infinity
        var x: CGFloat = 0, y: CGFloat = 0, rowHeight: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x + size.width > width, x > 0 {
                x = 0
                y += rowHeight + spacing
                rowHeight = 0
            }
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
        return CGSize(width: proposal.width ?? x, height: y + rowHeight)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var x = bounds.minX, y = bounds.minY, rowHeight: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x + size.width > bounds.maxX, x > bounds.minX {
                x = bounds.minX
                y += rowHeight + spacing
                rowHeight = 0
            }
            subview.place(at: CGPoint(x: x, y: y), proposal: .unspecified)
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
    }
}
