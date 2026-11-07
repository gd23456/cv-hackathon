import KshanaCore
import SwiftUI

/// What ran for this image, shown below it rather than on top of it.
struct RunReportView: View {
    let studio: Studio
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var shown = false

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            if let line = headline { Text(line).font(.subheadline.weight(.semibold)).monospacedDigit() }
            if let reason = studio.plan?.reason {
                Text(reason).font(.caption).foregroundStyle(.secondary)
            }
            if !studio.checks.isEmpty { promptChecks }
            if !studio.detections.isEmpty { objects }
            if let summary = studio.summary { steps(summary) }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 16)
        .padding(.top, 36)
        .padding(.bottom, 16)
        .glass(in: FolderShape(tabWidth: 100, tabHeight: 24))
        .folderTab("Report", height: 24)
    }

    private var headline: String? {
        if let summary = studio.summary {
            return String(format: "%d images in %.1f s. First image in %.0f ms.", summary.candidatesTried, summary.elapsedMs / 1000, summary.firstImageMs)
        }
        return studio.phase == .generating ? "Searching: \(studio.triedCount) images so far" : nil
    }

    private var promptChecks: some View {
        let verifiable = studio.checks.filter { $0.state != .notVerifiable }
        let passed = verifiable.filter { $0.state == .passed }.count
        return section("Prompt check", note: verifiable.isEmpty ? nil : "\(passed) of \(verifiable.count) passed") {
            ForEach(studio.checks) { check in
                row(check.title, value: stateText(check.state), symbol: symbol(check.state), tint: tint(check.state))
            }
        }
    }

    private var objects: some View {
        section("Objects found", note: nil) {
            Text(studio.detections.prefix(8).map { "\($0.label) \(Int($0.confidence * 100))%" }.joined(separator: ", "))
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }

    private func steps(_ summary: SessionSummary) -> some View {
        let t = summary.best.timings
        let items: [(String, String, Bool)] = [
            ("Prompt filter", "Passed", true),
            ("Text encoder", ms(t.textMs), true),
            ("Image model, 1 step", ms(t.unetMs), true),
            ("Decoder", ms(t.decodeMs), true),
            ("Safety check, every image", summary.safetyRejections == 0 ? ms(summary.best.safetyMs) : "\(summary.safetyRejections) rejected", true),
            ("Object and prompt checks", ms(summary.best.verifyMs), summary.best.verifyMs > 0),
            ("Aesthetics ranking", summary.finalists > 1 ? "\(summary.finalists) finalists" : "Not needed", summary.finalists > 1),
        ] + summary.stages.map { ($0.name, $0.applied ? ms($0.milliseconds) : "Skipped", $0.applied) }

        return section("What ran", note: studio.acceleration?.isAccelerated == false ? "CPU" : "Neural Engine, no network") {
            ForEach(Array(items.enumerated()), id: \.offset) { index, item in
                row(item.0, value: item.1, symbol: item.2 ? "checkmark" : "minus", tint: item.2 ? .secondary : .secondary.opacity(0.5))
                    .opacity(shown || reduceMotion ? 1 : 0)
                    .offset(y: shown || reduceMotion ? 0 : 6)
                    .animation(reduceMotion ? nil : .snappy(duration: 0.3).delay(Double(index) * 0.035), value: shown)
            }
        }
        .id(studio.completedRuns)
        .onAppear {
            shown = false
            Task { @MainActor in shown = true }
        }
    }

    private func section<Content: View>(_ title: String, note: String?, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(title).font(.footnote.weight(.semibold))
                Spacer()
                if let note { Text(note).font(.caption).foregroundStyle(.secondary) }
            }
            content()
        }
    }

    private func row(_ title: String, value: String, symbol: String, tint: Color) -> some View {
        HStack(spacing: 8) {
            Image(systemName: symbol).font(.caption2.weight(.bold)).foregroundStyle(tint).frame(width: 14)
            Text(title).font(.footnote)
            Spacer(minLength: 8)
            Text(value).font(.footnote.monospacedDigit()).foregroundStyle(.secondary)
        }
    }

    private func ms(_ value: Double) -> String { String(format: "%.0f ms", value) }

    private func symbol(_ state: CheckResult.State) -> String {
        switch state {
        case .passed: "checkmark"
        case .failed: "xmark"
        case .notVerifiable: "minus"
        }
    }

    private func tint(_ state: CheckResult.State) -> Color {
        switch state {
        case .passed: Color(red: 0.35, green: 0.78, blue: 0.5)
        case .failed: Color(red: 0.95, green: 0.42, blue: 0.4)
        case .notVerifiable: .secondary
        }
    }

    private func stateText(_ state: CheckResult.State) -> String {
        switch state {
        case .passed: "Found"
        case .failed: "Not found"
        case .notVerifiable: "Can't check"
        }
    }
}
