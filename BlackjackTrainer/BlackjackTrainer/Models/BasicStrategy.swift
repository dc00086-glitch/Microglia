import SwiftUI

/// The five things a player can choose to do.
enum Action: String, CaseIterable, Codable {
    case hit, stand, double, split, surrender

    var title: String {
        switch self {
        case .hit:       return "Hit"
        case .stand:     return "Stand"
        case .double:    return "Double"
        case .split:     return "Split"
        case .surrender: return "Surrender"
        }
    }
}

/// The "book" move for a chart cell, including conditional moves that depend on
/// whether doubling / surrender are allowed for the situation.
enum BookMove {
    case hit
    case stand
    case doubleElseHit          // D  – double if allowed, otherwise hit
    case doubleElseStand        // Ds – double if allowed, otherwise stand
    case split                  // P
    case splitIfDASElseHit      // P* – split if double-after-split allowed, else hit
    case surrenderElseHit       // Rh – surrender if allowed, otherwise hit
    case surrenderElseStand     // Rs – surrender if allowed, otherwise stand

    /// Two-letter code shown in the strategy chart.
    var code: String {
        switch self {
        case .hit:                return "H"
        case .stand:              return "S"
        case .doubleElseHit:      return "D"
        case .doubleElseStand:    return "Ds"
        case .split:              return "P"
        case .splitIfDASElseHit:  return "P*"
        case .surrenderElseHit:   return "Rh"
        case .surrenderElseStand: return "Rs"
        }
    }

    var color: Color {
        switch self {
        case .hit:                                  return Color(red: 0.86, green: 0.30, blue: 0.27) // red = hit
        case .stand:                                return Color(red: 0.95, green: 0.80, blue: 0.25) // yellow = stand
        case .doubleElseHit, .doubleElseStand:      return Color(red: 0.36, green: 0.66, blue: 0.92) // blue = double
        case .split, .splitIfDASElseHit:            return Color(red: 0.40, green: 0.74, blue: 0.45) // green = split
        case .surrenderElseHit, .surrenderElseStand:return Color(red: 0.62, green: 0.55, blue: 0.85) // purple = surrender
        }
    }
}

/// The resolved recommendation for a specific live situation.
struct StrategyDecision {
    let action: Action
    let book: BookMove
    let explanation: String
}

/// Stateless basic-strategy engine for 4–8 deck games. All tables follow the
/// standard published charts (e.g. Wizard of Odds) with H17/S17 and DAS
/// variations applied.
enum BasicStrategy {

    // MARK: Chart lookups (dealer is the upcard pip: 2...10, or 11 for Ace)

    static func hardMove(total: Int, dealer d: Int, h17: Bool) -> BookMove {
        switch total {
        case ...8:  return .hit
        case 9:     return (3...6).contains(d) ? .doubleElseHit : .hit
        case 10:    return (2...9).contains(d) ? .doubleElseHit : .hit
        case 11:    // S17: hit vs Ace. H17: double vs everything.
            if h17 { return .doubleElseHit }
            return d == 11 ? .hit : .doubleElseHit
        case 12:    return (4...6).contains(d) ? .stand : .hit
        case 13, 14: return (2...6).contains(d) ? .stand : .hit
        case 15:
            if (2...6).contains(d) { return .stand }
            if d == 10 { return .surrenderElseHit }
            if d == 11 && h17 { return .surrenderElseHit }   // 15 vs A surrenders only in H17
            return .hit
        case 16:
            if (2...6).contains(d) { return .stand }
            if d >= 9 { return .surrenderElseHit }            // 9, 10, A
            return .hit                                       // 7, 8
        default:    return .stand                             // 17+
        }
    }

    static func softMove(total: Int, dealer d: Int, h17: Bool) -> BookMove {
        switch total {
        case 13, 14: return (5...6).contains(d) ? .doubleElseHit : .hit       // A2, A3
        case 15, 16: return (4...6).contains(d) ? .doubleElseHit : .hit       // A4, A5
        case 17:     return (3...6).contains(d) ? .doubleElseHit : .hit       // A6
        case 18:                                                              // A7
            if (3...6).contains(d) { return .doubleElseStand }
            if h17 && d == 2 { return .doubleElseStand }
            if d == 2 || d == 7 || d == 8 { return .stand }
            return .hit                                                       // 9, 10, A
        case 19:                                                              // A8
            if h17 && d == 6 { return .doubleElseStand }
            return .stand
        default:     return .stand                                            // A9, A10
        }
    }

    static func pairMove(rank: Rank, dealer d: Int, das: Bool, h17: Bool) -> BookMove {
        let p = rank.pip
        if rank == .ace { return .split }                                     // A,A always
        switch p {
        case 10: return .stand                                                // never split tens
        case 9:  return (d == 7 || d == 10 || d == 11) ? .stand : .split      // split 2-6, 8-9
        case 8:  return .split                                                // always
        case 7:  return (2...7).contains(d) ? .split : .hit
        case 6:  return das ? ((2...6).contains(d) ? .split : .hit)
                            : ((3...6).contains(d) ? .split : .hit)
        case 5:  return (2...9).contains(d) ? .doubleElseHit : .hit           // treat as hard 10
        case 4:  return (5...6).contains(d) ? .splitIfDASElseHit : .hit
        default: // 2s and 3s
                 return das ? ((2...7).contains(d) ? .split : .hit)
                            : ((4...7).contains(d) ? .split : .hit)
        }
    }

    /// The "book" move for a two-card starting hand, ignoring how many cards are
    /// actually in play. Used to render the chart cells.
    static func bookMove(for hand: Hand, dealer: Rank, rules: RuleSet) -> BookMove {
        let d = dealer.pip
        if let pr = hand.pairRank {
            return pairMove(rank: pr, dealer: d, das: rules.doubleAfterSplit, h17: rules.dealerHitsSoft17)
        }
        if hand.isSoft {
            return softMove(total: hand.total, dealer: d, h17: rules.dealerHitsSoft17)
        }
        return hardMove(total: hand.total, dealer: d, h17: rules.dealerHitsSoft17)
    }

    // MARK: Resolution for a live hand

    /// Resolve the book move into an actually-legal action given what the player
    /// can do right now (cards in hand, surrender offered, etc.).
    static func recommendation(player hand: Hand, dealer: Rank, rules: RuleSet) -> StrategyDecision {
        let d = dealer.pip
        let twoCards = hand.cards.count == 2
        let canDouble = twoCards
        let canSurrender = rules.surrenderAllowed && twoCards
        let canSplit = twoCards && hand.isPair

        let book = bookMove(for: hand, dealer: dealer, rules: rules)
        let action = resolve(book, canDouble: canDouble, canSplit: canSplit,
                             canSurrender: canSurrender, das: rules.doubleAfterSplit,
                             hand: hand, dealer: dealer, rules: rules)
        let why = explain(book: book, action: action, hand: hand, dealer: dealer, rules: rules)
        return StrategyDecision(action: action, book: book, explanation: why)
    }

    private static func resolve(_ book: BookMove, canDouble: Bool, canSplit: Bool,
                                canSurrender: Bool, das: Bool,
                                hand: Hand, dealer: Rank, rules: RuleSet) -> Action {
        switch book {
        case .hit:                return .hit
        case .stand:              return .stand
        case .doubleElseHit:      return canDouble ? .double : .hit
        case .doubleElseStand:    return canDouble ? .double : .stand
        case .surrenderElseHit:   return canSurrender ? .surrender : .hit
        case .surrenderElseStand: return canSurrender ? .surrender : .stand
        case .split:
            return canSplit ? .split : fallbackForUnsplit(hand: hand, dealer: dealer, rules: rules, canDouble: canDouble)
        case .splitIfDASElseHit:
            return (canSplit && das) ? .split : .hit
        }
    }

    /// If a pair can't be split (more than two cards in play), fall back to the
    /// hard/soft total decision.
    private static func fallbackForUnsplit(hand: Hand, dealer: Rank, rules: RuleSet, canDouble: Bool) -> Action {
        let d = dealer.pip
        let book = hand.isSoft
            ? softMove(total: hand.total, dealer: d, h17: rules.dealerHitsSoft17)
            : hardMove(total: hand.total, dealer: d, h17: rules.dealerHitsSoft17)
        switch book {
        case .stand, .doubleElseStand:           return canDouble ? .double : .stand
        case .doubleElseHit:                      return canDouble ? .double : .hit
        case .surrenderElseStand:                 return .stand
        default:                                  return .hit
        }
    }

    // MARK: Plain-language explanations

    private static func explain(book: BookMove, action: Action, hand: Hand, dealer: Rank, rules: RuleSet) -> String {
        let dealerStrong = (7...11).contains(dealer.pip)
        let dealerWeak = (2...6).contains(dealer.pip)
        let dLabel = dealer == .ace ? "an Ace" : "a \(dealer.label)"

        switch book {
        case .split:
            if hand.pairRank == .ace { return "Always split Aces — two fresh hands each starting with 11 is far better than a single soft/hard 12." }
            if hand.pairRank == .eight { return "Always split 8s. A hard 16 is the worst hand in blackjack; two hands starting with 8 are much stronger." }
            return "Split. Against \(dLabel) this pair wins more as two separate hands than played as one total of \(hand.total)."
        case .splitIfDASElseHit:
            return "Split only because you can double afterward (DAS). Without DAS, just hit."
        case .doubleElseHit:
            return "Double down. You have a strong starting hand and the dealer shows \(dLabel), so press your bet. (If you can't double — e.g. after hitting — just hit.)"
        case .doubleElseStand:
            return "Double for value against \(dLabel). If doubling isn't allowed, stand."
        case .surrenderElseHit:
            return "Surrender. Against \(dLabel) your \(hand.describe.lowercased()) loses too often — give up half the bet. If surrender isn't offered, hit."
        case .surrenderElseStand:
            return "Surrender if allowed; otherwise stand."
        case .stand:
            if dealerWeak { return "Stand. The dealer's \(dLabel) is a weak ('bust') card — let them draw and risk busting instead of risking it yourself." }
            return "Stand. Hitting risks busting and your total is already competitive."
        case .hit:
            if dealerStrong { return "Hit. The dealer's \(dLabel) is strong, so a weak total like \(hand.total) must be improved." }
            return "Hit. Your total is low enough that drawing is the better play."
        }
    }
}
