import SwiftUI

@main
struct KshanaApp: App {
    @State private var studio = Studio()
    @State private var tab = CommandLine.arguments.contains("--tab-benchmarks") ? 2 : CommandLine.arguments.contains("--tab-inside") ? 1 : 0
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            TabView(selection: $tab) {
                Tab("Create", systemImage: "sparkles", value: 0) { CreateView(studio: studio) }
                Tab("Inside", systemImage: "cpu", value: 1) { InsideView(studio: studio) }
                Tab("Benchmarks", systemImage: "chart.xyaxis.line", value: 2) { BenchmarksView(studio: studio) }
            }
            .tint(Color.kshanaAccent)
            .preferredColorScheme(.dark)
            .task {
                if CommandLine.arguments.contains("--plan") {
                    await studio.runPlanReport()
                    exit(0)
                }
                await studio.prepare()
                if CommandLine.arguments.contains("--live-test") {
                    await studio.runLiveTest()
                }
                if CommandLine.arguments.contains("--ui-shots") {
                    await studio.saveInterfaceShots()
                }
                if CommandLine.arguments.contains("--ui-test") {
                    await studio.runInterfaceTest()
                }
                if CommandLine.arguments.contains("--bench") {
                    await studio.runCommandLineBenchmark()
                    exit(0)
                }
            }
            .onChange(of: scenePhase) { _, phase in
                if phase == .background { studio.cancelForBackground() }
                if phase == .active { studio.refreshDevice() }
            }
            .onReceive(NotificationCenter.default.publisher(for: UIApplication.didReceiveMemoryWarningNotification)) { _ in
                studio.handleMemoryWarning()
            }
        }
    }
}
