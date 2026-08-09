import SwiftUI

/// Tracks correct/total answers for the strategy trainer and the counting drill,
/// persisting a simple running tally.
final class TrainerStats: ObservableObject {
    @Published private(set) var strategyCorrect = 0
    @Published private(set) var strategyTotal = 0
    @Published private(set) var streak = 0
    @Published private(set) var bestStreak = 0

    @Published private(set) var countingCorrect = 0
    @Published private(set) var countingTotal = 0

    private let key = "blackjack.stats.v1"

    init() { load() }

    var strategyAccuracy: Double {
        strategyTotal == 0 ? 0 : Double(strategyCorrect) / Double(strategyTotal)
    }

    var countingAccuracy: Double {
        countingTotal == 0 ? 0 : Double(countingCorrect) / Double(countingTotal)
    }

    func recordStrategy(correct: Bool) {
        strategyTotal += 1
        if correct {
            strategyCorrect += 1
            streak += 1
            bestStreak = max(bestStreak, streak)
        } else {
            streak = 0
        }
        save()
    }

    func recordCounting(correct: Bool) {
        countingTotal += 1
        if correct { countingCorrect += 1 }
        save()
    }

    func resetStrategy() {
        strategyCorrect = 0; strategyTotal = 0; streak = 0; bestStreak = 0
        save()
    }

    func resetCounting() {
        countingCorrect = 0; countingTotal = 0
        save()
    }

    // MARK: Persistence
    private struct Snapshot: Codable {
        var sc, st, streak, best, cc, ct: Int
    }

    private func save() {
        let snap = Snapshot(sc: strategyCorrect, st: strategyTotal, streak: streak,
                            best: bestStreak, cc: countingCorrect, ct: countingTotal)
        if let data = try? JSONEncoder().encode(snap) {
            UserDefaults.standard.set(data, forKey: key)
        }
    }

    private func load() {
        guard let data = UserDefaults.standard.data(forKey: key),
              let snap = try? JSONDecoder().decode(Snapshot.self, from: data) else { return }
        strategyCorrect = snap.sc; strategyTotal = snap.st
        streak = snap.streak; bestStreak = snap.best
        countingCorrect = snap.cc; countingTotal = snap.ct
    }
}
