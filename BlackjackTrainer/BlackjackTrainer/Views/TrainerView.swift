import SwiftUI

struct TrainerView: View {
    @EnvironmentObject var settings: AppSettings
    @EnvironmentObject var stats: TrainerStats

    @State private var shoe = Shoe(decks: 6)
    @State private var player = Hand()
    @State private var dealer = Card(rank: .ace, suit: .spades)
    @State private var feedback: Feedback?

    struct Feedback {
        let correct: Bool
        let chosen: Action
        let decision: StrategyDecision
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    scoreBar

                    // Dealer
                    VStack(spacing: 6) {
                        Text("DEALER SHOWS").font(.caption).foregroundStyle(.secondary)
                        HStack(spacing: 6) {
                            CardView(card: dealer, size: 70)
                            CardView(card: dealer, size: 70, faceDown: true)
                        }
                    }

                    Divider().padding(.horizontal, 60)

                    // Player
                    VStack(spacing: 6) {
                        Text("YOUR HAND").font(.caption).foregroundStyle(.secondary)
                        HStack(spacing: 6) {
                            ForEach(player.cards) { CardView(card: $0, size: 70) }
                        }
                        Text(player.describe).font(.headline)
                    }

                    if let fb = feedback {
                        feedbackView(fb)
                    } else {
                        actionButtons
                    }
                }
                .padding()
            }
            .navigationTitle("Trainer")
            .onAppear { if player.cards.isEmpty { deal() } }
        }
    }

    // MARK: Subviews

    private var scoreBar: some View {
        HStack {
            stat("Accuracy", String(format: "%.0f%%", stats.strategyAccuracy * 100))
            Spacer()
            stat("Streak", "\(stats.streak)")
            Spacer()
            stat("Best", "\(stats.bestStreak)")
            Spacer()
            stat("Hands", "\(stats.strategyTotal)")
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private func stat(_ label: String, _ value: String) -> some View {
        VStack(spacing: 2) {
            Text(value).font(.headline.monospacedDigit())
            Text(label).font(.caption2).foregroundStyle(.secondary)
        }
    }

    private var availableActions: [Action] {
        var acts: [Action] = [.hit, .stand]
        if player.cards.count == 2 {
            acts.append(.double)
            if player.isPair { acts.append(.split) }
            if settings.rules.surrenderAllowed { acts.append(.surrender) }
        }
        return acts
    }

    private var actionButtons: some View {
        VStack(spacing: 10) {
            Text("What's the correct play?").font(.subheadline).foregroundStyle(.secondary)
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                ForEach(availableActions, id: \.self) { act in
                    Button { choose(act) } label: {
                        Text(act.title)
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 14)
                            .background(Color.accentColor.opacity(0.15))
                            .clipShape(RoundedRectangle(cornerRadius: 10))
                    }
                }
            }
        }
    }

    private func feedbackView(_ fb: Feedback) -> some View {
        VStack(spacing: 12) {
            HStack(spacing: 8) {
                Image(systemName: fb.correct ? "checkmark.circle.fill" : "xmark.circle.fill")
                    .foregroundColor(fb.correct ? .green : .red)
                Text(fb.correct ? "Correct!" : "You chose \(fb.chosen.title) — correct is \(fb.decision.action.title)")
                    .font(.headline)
            }
            Text(fb.decision.explanation)
                .font(.callout)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
            Button { deal() } label: {
                Text("Next Hand")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .background(Color.accentColor)
                    .foregroundColor(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
            }
        }
        .padding(.top, 4)
    }

    // MARK: Logic

    private func choose(_ act: Action) {
        let decision = BasicStrategy.recommendation(player: player, dealer: dealer.rank, rules: settings.rules)
        let correct = (act == decision.action)
        stats.recordStrategy(correct: correct)
        feedback = Feedback(correct: correct, chosen: act, decision: decision)
    }

    private func deal() {
        if shoe.decks != settings.rules.decks {
            shoe = Shoe(decks: settings.rules.decks)
        }
        if shoe.cardsRemaining < 15 { shoe.reset() }
        // Re-deal until the player's two cards leave a real decision (skip naturals).
        var p = Hand()
        var up = Card(rank: .ace, suit: .spades)
        repeat {
            p = Hand([shoe.deal(), shoe.deal()])
            up = shoe.deal()
        } while p.isBlackjack
        player = p
        dealer = up
        feedback = nil
    }
}
