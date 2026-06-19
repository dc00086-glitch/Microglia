import SwiftUI

/// Holds the table rules and persists them between launches.
final class AppSettings: ObservableObject {
    @Published var rules: RuleSet {
        didSet { save() }
    }

    private let key = "blackjack.ruleset.v1"

    init() {
        if let data = UserDefaults.standard.data(forKey: key),
           let decoded = try? JSONDecoder().decode(RuleSet.self, from: data) {
            rules = decoded
        } else {
            rules = .default
        }
    }

    private func save() {
        if let data = try? JSONEncoder().encode(rules) {
            UserDefaults.standard.set(data, forKey: key)
        }
    }
}
