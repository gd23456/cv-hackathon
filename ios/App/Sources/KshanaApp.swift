import SwiftUI

@main
struct KshanaApp: App {
    @State private var studio = Studio()

    var body: some Scene {
        WindowGroup {
            CreateView(studio: studio)
                .task { await studio.prepare() }
        }
    }
}
