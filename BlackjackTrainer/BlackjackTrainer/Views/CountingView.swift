import SwiftUI

struct CountingView: View {
    enum Mode: String, CaseIterable, Identifiable {
        case tags = "Card Tags"
        case running = "Running Count"
        var id: String { rawValue }
    }

    @State private var mode: Mode = .tags

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                Picker("Mode", selection: $mode) {
                    ForEach(Mode.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .padding(.horizontal)

                switch mode {
                case .tags:    TagDrill()
                case .running: RunningCountDrill()
                }
                Spacer()
            }
            .padding(.top, 8)
            .navigationTitle("Card Counting")
        }
    }
}

// MARK: - Tag drill: classify each card +1 / 0 / −1

private struct TagDrill: View {
    @EnvironmentObject var stats: TrainerStats
    @State private var card = Card(rank: .ace, suit: .spades)
    @State private var result: Bool?

    var body: some View {
        VStack(spacing: 20) {
            Text("Hi-Lo: 2–6 = +1 · 7–9 = 0 · 10/J/Q/K/A = −1")
                .font(.footnote).foregroundStyle(.secondary)

            CardView(card: card, size: 110)

            if let r = result {
                Text(r ? "Correct (\(tagLabel(card.rank.hiLoValue)))"
                       : "Nope — \(card.rank.label) is \(tagLabel(card.rank.hiLoValue))")
                    .font(.headline)
                    .foregroundColor(r ? .green : .red)
            } else {
                Text("Tag this card").font(.subheadline).foregroundStyle(.secondary)
            }

            HStack(spacing: 12) {
                tagButton(+1, "+1", .green)
                tagButton(0, "0", .gray)
                tagButton(-1, "−1", .red)
            }

            Text(String(format: "Accuracy %.0f%%  ·  %d cards",
                        stats.countingAccuracy * 100, stats.countingTotal))
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding()
        .onAppear(perform: next)
    }

    private func tagButton(_ value: Int, _ label: String, _ color: Color) -> some View {
        Button {
            guard result == nil else { return }
            let correct = (value == card.rank.hiLoValue)
            stats.recordCounting(correct: correct)
            result = correct
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.7) { next() }
        } label: {
            Text(label)
                .font(.title2.bold())
                .frame(width: 80, height: 56)
                .background(color.opacity(0.2))
                .clipShape(RoundedRectangle(cornerRadius: 12))
        }
    }

    private func next() {
        card = Card(rank: Rank.allCases.randomElement()!, suit: Suit.allCases.randomElement()!)
        result = nil
    }

    private func tagLabel(_ v: Int) -> String { v > 0 ? "+1" : (v < 0 ? "−1" : "0") }
}

// MARK: - Running count drill: flip a sequence, then state the count

private struct RunningCountDrill: View {
    @EnvironmentObject var stats: TrainerStats
    @EnvironmentObject var settings: AppSettings

    enum Phase { case idle, dealing, answering, revealed }

    @State private var phase: Phase = .idle
    @State private var shoe = Shoe(decks: 6)
    @State private var sequence: [Card] = []
    @State private var index = 0
    @State private var length = 15
    @State private var guessRunning = 0
    @State private var guessTrue = 0

    private var seenRunning: Int { sequence.prefix(index).reduce(0) { $0 + $1.rank.hiLoValue } }
    private var decksRemaining: Double {
        max(0.5, Double(settings.rules.decks * 52 - index) / 52.0)
    }
    private var actualTrue: Int { Int((Double(seenRunning) / decksRemaining).rounded()) }

    var body: some View {
        VStack(spacing: 18) {
            switch phase {
            case .idle:
                VStack(spacing: 14) {
                    Text("Flip through cards one at a time and keep a running count in your head. At the end, enter the running count and true count.")
                        .font(.footnote).foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                    Stepper("Cards per round: \(length)", value: $length, in: 5...52, step: 5)
                    Button("Start Round") { start() }
                        .buttonStyle(.borderedProminent)
                }
                .padding()

            case .dealing:
                VStack(spacing: 16) {
                    Text("Card \(index) of \(sequence.count)")
                        .font(.caption).foregroundStyle(.secondary)
                    if index > 0 {
                        CardView(card: sequence[index - 1], size: 120)
                    } else {
                        RoundedRectangle(cornerRadius: 12).fill(Color(.tertiarySystemBackground))
                            .frame(width: 120, height: 168)
                            .overlay(Text("Tap Next").foregroundStyle(.secondary))
                    }
                    Button(index < sequence.count ? "Next Card" : "Done — Enter Count") {
                        if index < sequence.count { index += 1 }
                        else { phase = .answering }
                    }
                    .buttonStyle(.borderedProminent)
                }
                .padding()

            case .answering:
                VStack(spacing: 16) {
                    Text("\(sequence.count) cards shown.").font(.headline)
                    Stepper("Running count: \(guessRunning)", value: $guessRunning, in: -40...40)
                    Stepper("True count: \(guessTrue)", value: $guessTrue, in: -40...40)
                    Text(String(format: "(%.1f decks remaining)", decksRemaining))
                        .font(.caption).foregroundStyle(.secondary)
                    Button("Check") {
                        let correct = (guessRunning == seenRunning)
                        stats.recordCounting(correct: correct)
                        phase = .revealed
                    }
                    .buttonStyle(.borderedProminent)
                }
                .padding()

            case .revealed:
                VStack(spacing: 14) {
                    resultRow("Running count", guess: guessRunning, actual: seenRunning)
                    resultRow("True count", guess: guessTrue, actual: actualTrue)
                    Text("True count = running count (\(seenRunning)) ÷ decks remaining (\(String(format: "%.1f", decksRemaining))).")
                        .font(.caption).foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                    Button("New Round") { phase = .idle }
                        .buttonStyle(.borderedProminent)
                }
                .padding()
            }
        }
    }

    private func resultRow(_ label: String, guess: Int, actual: Int) -> some View {
        let ok = guess == actual
        return HStack {
            Image(systemName: ok ? "checkmark.circle.fill" : "xmark.circle.fill")
                .foregroundColor(ok ? .green : .red)
            Text(label)
            Spacer()
            Text("you \(guess) · actual \(actual)")
                .font(.subheadline.monospacedDigit())
                .foregroundStyle(.secondary)
        }
        .font(.headline)
    }

    private func start() {
        shoe = Shoe(decks: settings.rules.decks)
        sequence = shoe.deal(length)
        index = 0
        guessRunning = 0
        guessTrue = 0
        phase = .dealing
    }
}
