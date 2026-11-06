import SwiftUI

@main
struct KshanaApp: App {
    @State private var studio = Studio()

    var body: some Scene {
        WindowGroup {
            CreateView(studio: studio)
                .task {
                    if CommandLine.arguments.contains("--plan") {
                        await studio.runPlanReport()
                        exit(0)
                    }
                    await studio.prepare()
                    if CommandLine.arguments.contains("--bench") {
                        await studio.runBenchmark()
                        exit(0)
                    }
                }
        }
    }
}
