import SwiftUI

struct StrategyChartView: View {
    @EnvironmentObject var settings: AppSettings
    @State private var selection: ChartSelection?

    /// Dealer upcards across the top: 2...10, then Ace.
    private let dealerRanks: [Rank] = [.two, .three, .four, .five, .six, .seven, .eight, .nine, .ten, .ace]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    Text("Tap any cell to see the play and the reasoning. Rows are your hand, columns are the dealer's upcard.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .padding(.horizontal)

                    section("Hard Totals", rows: hardRows)
                    section("Soft Totals (one Ace)", rows: softRows)
                    section("Pairs", rows: pairRows)

                    Legend()
                        .padding(.horizontal)
                        .padding(.bottom, 24)
                }
                .padding(.top, 8)
            }
            .navigationTitle("Strategy Chart")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Text(settings.rules.summary)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            .sheet(item: $selection) { sel in
                DecisionDetailSheet(hand: sel.hand, dealer: sel.dealer)
                    .presentationDetents([.medium])
            }
        }
    }

    // MARK: Row definitions

    private var hardRows: [ChartRow] {
        (8...17).map { total in
            ChartRow(label: total == 17 ? "17+" : "\(total)",
                     hand: { Self.hardHand(total: total) })
        }
    }

    private var softRows: [ChartRow] {
        (13...20).map { total in
            ChartRow(label: "A-\(total - 11)",
                     hand: { Hand([Card(rank: .ace, suit: .spades),
                                   Card(rank: Rank(rawValue: total - 11)!, suit: .hearts)]) })
        }
    }

    private var pairRows: [ChartRow] {
        let ranks: [Rank] = [.two, .three, .four, .five, .six, .seven, .eight, .nine, .ten, .ace]
        return ranks.map { r in
            ChartRow(label: "\(r.label)-\(r.label)",
                     hand: { Hand([Card(rank: r, suit: .spades), Card(rank: r, suit: .clubs)]) })
        }
    }

    // MARK: Section grid

    @ViewBuilder
    private func section(_ title: String, rows: [ChartRow]) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.headline)
                .padding(.horizontal)

            ScrollView(.horizontal, showsIndicators: false) {
                VStack(spacing: 2) {
                    // Header row
                    HStack(spacing: 2) {
                        cellFrame { Text("").font(.caption2) }
                        ForEach(dealerRanks, id: \.self) { d in
                            cellFrame {
                                Text(d.label)
                                    .font(.caption2.bold())
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                    ForEach(rows) { row in
                        HStack(spacing: 2) {
                            cellFrame {
                                Text(row.label)
                                    .font(.caption2.bold())
                                    .foregroundStyle(.secondary)
                            }
                            ForEach(dealerRanks, id: \.self) { d in
                                let hand = row.hand()
                                let book = BasicStrategy.bookMove(for: hand, dealer: d, rules: settings.rules)
                                Button {
                                    selection = ChartSelection(hand: hand, dealer: d)
                                } label: {
                                    cellFrame {
                                        Text(book.code)
                                            .font(.caption2.bold())
                                            .foregroundColor(.black.opacity(0.85))
                                    }
                                    .background(book.color)
                                    .clipShape(RoundedRectangle(cornerRadius: 3))
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    }
                }
                .padding(.horizontal)
            }
        }
    }

    private func cellFrame<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        content()
            .frame(width: 30, height: 26)
    }

    // Build a representative hard hand for a given total (never a pair, no Ace).
    static func hardHand(total: Int) -> Hand {
        if total <= 11 {
            let first = total - 2          // 6..9
            return Hand([Card(rank: Rank(rawValue: first)!, suit: .spades),
                         Card(rank: .two, suit: .hearts)])
        } else {
            return Hand([Card(rank: .ten, suit: .spades),
                         Card(rank: Rank(rawValue: total - 10)!, suit: .hearts)])
        }
    }
}

struct ChartRow: Identifiable {
    let id = UUID()
    let label: String
    let hand: () -> Hand
}

struct ChartSelection: Identifiable {
    let id = UUID()
    let hand: Hand
    let dealer: Rank
}

/// Bottom-sheet shown when a chart cell (or trainer hand) is inspected.
struct DecisionDetailSheet: View {
    @EnvironmentObject var settings: AppSettings
    let hand: Hand
    let dealer: Rank

    var body: some View {
        let decision = BasicStrategy.recommendation(player: hand, dealer: dealer, rules: settings.rules)
        return VStack(spacing: 18) {
            Capsule().fill(.secondary.opacity(0.4)).frame(width: 40, height: 5).padding(.top, 8)

            HStack(spacing: 24) {
                VStack(spacing: 6) {
                    Text("You").font(.caption).foregroundStyle(.secondary)
                    HStack(spacing: 4) {
                        ForEach(hand.cards) { CardView(card: $0, size: 44) }
                    }
                    Text(hand.describe).font(.subheadline.bold())
                }
                VStack(spacing: 6) {
                    Text("Dealer").font(.caption).foregroundStyle(.secondary)
                    HStack(spacing: 4) {
                        CardView(card: Card(rank: dealer, suit: .spades), size: 44)
                        CardView(card: Card(rank: .ace, suit: .clubs), size: 44, faceDown: true)
                    }
                    Text("Shows \(dealer.label)").font(.subheadline.bold())
                }
            }

            Text(decision.action.title.uppercased())
                .font(.title2.bold())
                .padding(.horizontal, 20).padding(.vertical, 8)
                .background(decision.book.color)
                .foregroundColor(.black.opacity(0.85))
                .clipShape(Capsule())

            Text(decision.explanation)
                .font(.callout)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
                .padding(.horizontal)

            Spacer()
        }
        .padding(.bottom)
    }
}

struct Legend: View {
    private let items: [(String, BookMove)] = [
        ("Hit", .hit), ("Stand", .stand), ("Double", .doubleElseHit),
        ("Split", .split), ("Surrender", .surrenderElseHit)
    ]
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Legend").font(.headline)
            ForEach(items, id: \.0) { name, move in
                HStack(spacing: 8) {
                    RoundedRectangle(cornerRadius: 3).fill(move.color).frame(width: 22, height: 18)
                    Text("\(move.code) — \(name)").font(.caption)
                }
            }
            Text("Ds = double if allowed else stand · P* = split only with DAS · Rh/Rs = surrender else hit/stand")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .padding(.top, 2)
        }
    }
}
