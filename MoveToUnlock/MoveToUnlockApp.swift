import SwiftUI

@main
struct MoveToUnlockApp: App {
    @StateObject private var shield = ShieldManager()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(shield)
                .task {
                    await shield.requestAuthorization()
                }
        }
    }
}
