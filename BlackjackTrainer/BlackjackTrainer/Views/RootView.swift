import SwiftUI

struct RootView: View {
    var body: some View {
        TabView {
            LearnView()
                .tabItem { Label("Learn", systemImage: "book.fill") }

            StrategyChartView()
                .tabItem { Label("Strategy", systemImage: "tablecells.fill") }

            TrainerView()
                .tabItem { Label("Trainer", systemImage: "gamecontroller.fill") }

            CountingView()
                .tabItem { Label("Counting", systemImage: "number.circle.fill") }

            SettingsView()
                .tabItem { Label("Settings", systemImage: "gearshape.fill") }
        }
    }
}
