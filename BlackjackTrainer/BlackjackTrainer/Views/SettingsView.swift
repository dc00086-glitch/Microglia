import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var settings: AppSettings
    @EnvironmentObject var stats: TrainerStats

    var body: some View {
        NavigationStack {
            Form {
                Section("Table Rules") {
                    Stepper("Decks: \(settings.rules.decks)",
                            value: $settings.rules.decks, in: 1...8)
                    Toggle("Dealer hits soft 17 (H17)", isOn: $settings.rules.dealerHitsSoft17)
                    Toggle("Double after split (DAS)", isOn: $settings.rules.doubleAfterSplit)
                    Toggle("Late surrender offered", isOn: $settings.rules.surrenderAllowed)
                }

                Section("What these change") {
                    Text("These rules adjust the correct basic-strategy plays. The defaults (6 decks, stand on soft 17, DAS, late surrender) match a typical modern shoe game.")
                        .font(.footnote).foregroundStyle(.secondary)
                }

                Section("Your Progress") {
                    LabeledContent("Strategy accuracy",
                        value: String(format: "%.0f%% (%d hands)", stats.strategyAccuracy * 100, stats.strategyTotal))
                    LabeledContent("Best streak", value: "\(stats.bestStreak)")
                    LabeledContent("Counting accuracy",
                        value: String(format: "%.0f%% (%d cards)", stats.countingAccuracy * 100, stats.countingTotal))
                    Button("Reset strategy stats", role: .destructive) { stats.resetStrategy() }
                    Button("Reset counting stats", role: .destructive) { stats.resetCounting() }
                }

                Section {
                    Text("For learning only. Card counting is legal but casinos may ask advantage players to leave. Gamble responsibly.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Settings")
        }
    }
}
