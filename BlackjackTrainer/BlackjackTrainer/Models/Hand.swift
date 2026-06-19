import Foundation

/// A blackjack hand. Knows its best total, whether it is "soft" (an Ace still
/// counting as 11), whether it is a pair, and whether it is a natural blackjack.
struct Hand: Codable {
    var cards: [Card]

    init(_ cards: [Card] = []) { self.cards = cards }

    var count: Int { cards.count }

    /// Best total <= 21 if possible. Aces start at 11 and are demoted to 1.
    var total: Int {
        var sum = cards.reduce(0) { $0 + $1.rank.pip }
        var aces = cards.filter { $0.rank == .ace }.count
        while sum > 21 && aces > 0 {
            sum -= 10
            aces -= 1
        }
        return sum
    }

    /// True when an Ace is still counting as 11 (i.e. the hand is "soft").
    var isSoft: Bool {
        var sum = cards.reduce(0) { $0 + $1.rank.pip }
        var aces = cards.filter { $0.rank == .ace }.count
        while sum > 21 && aces > 0 {
            sum -= 10
            aces -= 1
        }
        return aces > 0
    }

    var isBust: Bool { total > 21 }

    /// Two cards of equal rank value (e.g. 10-K counts as a pair of tens).
    var isPair: Bool {
        cards.count == 2 && cards[0].rank.pip == cards[1].rank.pip
    }

    /// The rank used for pair decisions (so 10/J/Q/K all collapse to "ten").
    var pairRank: Rank? {
        guard isPair else { return nil }
        return cards[0].rank
    }

    var isBlackjack: Bool { cards.count == 2 && total == 21 }

    /// Human-readable description such as "Soft 18 (A-7)" or "Pair of 8s".
    var describe: String {
        if isBlackjack { return "Blackjack!" }
        if let pr = pairRank { return "Pair of \(pr.label)s" }
        if isSoft { return "Soft \(total)" }
        return "Hard \(total)"
    }
}
