import Foundation

/// A multi-deck shoe used by the trainer and the counting drills. Tracks the
/// running Hi-Lo count and how many cards remain so a true count can be derived.
struct Shoe {
    private(set) var cards: [Card] = []
    private(set) var dealt: [Card] = []
    let decks: Int

    init(decks: Int) {
        self.decks = decks
        reset()
    }

    mutating func reset() {
        cards = []
        dealt = []
        for _ in 0..<decks {
            for suit in Suit.allCases {
                for rank in Rank.allCases {
                    cards.append(Card(rank: rank, suit: suit))
                }
            }
        }
        cards.shuffle()
    }

    var cardsRemaining: Int { cards.count }

    var decksRemaining: Double {
        max(0.25, Double(cards.count) / 52.0)
    }

    /// Running Hi-Lo count of every card dealt so far.
    var runningCount: Int { dealt.reduce(0) { $0 + $1.rank.hiLoValue } }

    /// True count = running count / decks remaining (rounded to nearest int for drills).
    var trueCount: Int { Int((Double(runningCount) / decksRemaining).rounded()) }

    @discardableResult
    mutating func deal() -> Card {
        if cards.isEmpty { reset() }
        let c = cards.removeLast()
        dealt.append(c)
        return c
    }

    mutating func deal(_ n: Int) -> [Card] {
        (0..<n).map { _ in deal() }
    }
}
