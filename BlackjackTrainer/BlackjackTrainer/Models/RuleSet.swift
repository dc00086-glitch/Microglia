import Foundation

/// The table rules that change correct basic strategy. Defaults match the most
/// common modern shoe game: 4–8 decks, dealer stands on soft 17, double after
/// split allowed, late surrender offered.
struct RuleSet: Codable, Equatable {
    var decks: Int = 6
    /// Dealer hits soft 17 (H17) when true; stands on soft 17 (S17) when false.
    var dealerHitsSoft17: Bool = false
    /// Doubling allowed after splitting a pair (DAS).
    var doubleAfterSplit: Bool = true
    /// Late surrender offered.
    var surrenderAllowed: Bool = true

    static let `default` = RuleSet()

    /// Short tag for display, e.g. "6D · S17 · DAS · LS".
    var summary: String {
        var parts = ["\(decks)D"]
        parts.append(dealerHitsSoft17 ? "H17" : "S17")
        if doubleAfterSplit { parts.append("DAS") }
        if surrenderAllowed { parts.append("LS") }
        return parts.joined(separator: " · ")
    }
}
