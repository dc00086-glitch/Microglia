import SwiftUI

@main
struct BlackjackTrainerApp: App {
    @StateObject private var settings = AppSettings()
    @StateObject private var stats = TrainerStats()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(settings)
                .environmentObject(stats)
        }
    }
}
