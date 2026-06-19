import SwiftUI

/// A simple visual playing card.
struct CardView: View {
    let card: Card
    var size: CGFloat = 64
    var faceDown: Bool = false

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: size * 0.12)
                .fill(faceDown ? Color(red: 0.20, green: 0.32, blue: 0.55) : Color.white)
                .overlay(
                    RoundedRectangle(cornerRadius: size * 0.12)
                        .stroke(Color.black.opacity(0.18), lineWidth: 1)
                )
                .shadow(color: .black.opacity(0.18), radius: 2, x: 0, y: 1)

            if faceDown {
                Image(systemName: "suit.club.fill")
                    .font(.system(size: size * 0.4))
                    .foregroundColor(.white.opacity(0.65))
            } else {
                VStack(spacing: 0) {
                    Text(card.rank.label)
                        .font(.system(size: size * 0.42, weight: .bold, design: .rounded))
                    Text(card.suit.symbol)
                        .font(.system(size: size * 0.42))
                }
                .foregroundColor(card.suit.isRed ? .red : .black)
            }
        }
        .frame(width: size, height: size * 1.4)
    }
}

/// A small upcard used in chart headers / dealer display.
struct UpcardBadge: View {
    let rank: Rank
    var body: some View {
        Text(rank.label)
            .font(.system(.headline, design: .rounded).bold())
            .frame(width: 34, height: 46)
            .background(Color.white)
            .foregroundColor(.black)
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay(RoundedRectangle(cornerRadius: 6).stroke(.black.opacity(0.2)))
    }
}
