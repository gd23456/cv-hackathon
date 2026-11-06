import KshanaCore
import SwiftUI

struct CreateView: View {
    @Bindable var studio: Studio
    @FocusState private var promptFocused: Bool

    private let examples = [
        "a red apple to the left of a blue cup",
        "two dogs sitting on a couch",
        "a bicycle under a tree in the rain",
        "a cat above a yellow book",
    ]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
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
            .navigationTitle("Kshana")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Label(studio.acceleration?.isAccelerated == false ? "On this iPhone, CPU" : "On this iPhone", systemImage: "iphone.gen3")
                        .labelStyle(.titleAndIcon)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                if let image = studio.exportImage {
                    ToolbarItem(placement: .topBarTrailing) {
                        ShareLink(item: Image(uiImage: image), preview: SharePreview("AI-generated image", image: Image(uiImage: image)))
                    }
                }
            }
            .sensoryFeedback(.success, trigger: studio.completedRuns)
        }
    }

    private var canvas: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 16).fill(.quaternary.opacity(0.6))
            if let image = studio.image {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .clipShape(.rect(cornerRadius: 16))
                    .overlay(alignment: .bottomLeading) {
                        Text("AI generated")
                            .font(.caption2.weight(.medium))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(.ultraThinMaterial, in: .capsule)
                            .padding(10)
                    }
                    .accessibilityLabel("Generated image: \(studio.prompt)")
                    .contentTransition(.opacity)
            }
            canvasStatus
        }
        .aspectRatio(1, contentMode: .fit)
        .animation(.easeOut(duration: 0.18), value: studio.triedCount)
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
        case .generating where studio.image == nil || studio.triedCount == 0:
            ProgressView()
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
                ForEach(studio.checks) { check in CheckChip(check: check) }
            }
        }
    }

    private var composer: some View {
        VStack(alignment: .leading, spacing: 12) {
            TextField("Describe an image", text: $studio.prompt, axis: .vertical)
                .lineLimit(1...4)
                .focused($promptFocused)
                .submitLabel(.go)
                .onSubmit(generate)
                .padding(12)
                .background(.quaternary.opacity(0.6), in: .rect(cornerRadius: 12))
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
                Group {
                    if studio.phase == .generating {
                        Text(studio.triedCount > 0 ? "Checked \(studio.triedCount) images" : "Generating")
                    } else {
                        Text("Generate")
                    }
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 6)
            }
            .buttonStyle(.borderedProminent)
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
