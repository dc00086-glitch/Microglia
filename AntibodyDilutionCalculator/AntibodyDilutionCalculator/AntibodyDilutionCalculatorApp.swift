import SwiftUI

@main
struct AntibodyDilutionCalculatorApp: App {
    var body: some Scene {
        #if os(macOS)
        WindowGroup {
            ContentView()
                .frame(minWidth: 400, minHeight: 500)
        }
        #else
        WindowGroup {
            ContentView()
        }
        #endif
    }
}
