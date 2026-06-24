import Foundation

/// The four suits. Used only for display – strategy never depends on suit.
enum Suit: String, CaseIterable, Codable {
    case spades, hearts, diamonds, clubs

    var symbol: String {
        switch self {
        case .spades:   return "♠"
        case .hearts:   return "♥"
        case .diamonds: return "♦"
        case .clubs:    return "♣"
        }
    }

    var isRed: Bool { self == .hearts || self == .diamonds }
}

/// Card ranks. `pip` is the blackjack point value (Ace counts as 11 here; the
/// hand-evaluation logic demotes Aces to 1 as needed).
enum Rank: Int, CaseIterable, Codable, Comparable {
    case two = 2, three, four, five, six, seven, eight, nine, ten
    case jack, queen, king, ace

    /// Blackjack point value. Faces = 10, Ace = 11 (soft).
    var pip: Int {
        switch self {
        case .jack, .queen, .king: return 10
        case .ace:                 return 11
        default:                   return rawValue
        }
    }

    /// Short label used on the card face and in the strategy chart.
    var label: String {
        switch self {
        case .two:   return "2"
        case .three: return "3"
        case .four:  return "4"
        case .five:  return "5"
        case .six:   return "6"
        case .seven: return "7"
        case .eight: return "8"
        case .nine:  return "9"
        case .ten:   return "10"
        case .jack:  return "J"
        case .queen: return "Q"
        case .king:  return "K"
        case .ace:   return "A"
        }
    }

    /// Hi-Lo running-count tag: 2–6 = +1, 7–9 = 0, 10/face/Ace = −1.
    var hiLoValue: Int {
        switch pip {
        case 2...6:   return 1
        case 7...9:   return 0
        default:      return -1   // 10s, faces, Ace
        }
    }

    static func < (lhs: Rank, rhs: Rank) -> Bool { lhs.pip < rhs.pip }
}

/// A single playing card.
struct Card: Identifiable, Hashable, Codable {
    let rank: Rank
    let suit: Suit
    var id: String { "\(rank.label)\(suit.symbol)" }

    var label: String { "\(rank.label)\(suit.symbol)" }
}
