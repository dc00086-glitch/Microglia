import Foundation

struct Lesson: Identifiable {
    let id = UUID()
    let title: String
    let summary: String
    let body: String   // simple markdown-ish text rendered as paragraphs / bullets
}

enum LessonLibrary {
    static let all: [Lesson] = [
        Lesson(
            title: "The Goal & Hand Values",
            summary: "Beat the dealer without busting. How cards are counted.",
            body: """
            The goal of blackjack is to have a hand total **closer to 21 than the dealer** without going over 21 ("busting").

            • Number cards (2–10) are worth their face value.
            • Jack, Queen, King are each worth 10.
            • An Ace is worth **11 or 1**, whichever helps you more.

            A hand with an Ace counting as 11 is called a **soft** hand (e.g. A-6 = "soft 17"). If counting the Ace as 11 would bust you, it becomes 1 and the hand is **hard**.

            A **blackjack** (natural) is an Ace + any 10-value card on your first two cards. It usually pays **3:2** — avoid tables that pay 6:5, the house edge is much worse.
            """
        ),
        Lesson(
            title: "Hitting & Standing",
            summary: "When to take a card vs. keep your total.",
            body: """
            **Hit** = take another card. **Stand** = keep your current total.

            The dealer must follow fixed rules: they hit until 17+ and then stand (on a soft 17 they either hit or stand depending on the table — "H17" vs "S17").

            Key idea — the **dealer's upcard** drives everything:
            • Dealer shows **2–6** (weak/"bust cards"): the dealer busts often, so you stand on stiff totals (12–16) and let them take the risk.
            • Dealer shows **7–Ace** (strong): you must improve weak hands, so you hit more aggressively.

            Always assume the dealer's hidden card is a 10. That's why a dealer 6 looks like a likely 16 (a bust waiting to happen), while a dealer 10 looks like a made 20.
            """
        ),
        Lesson(
            title: "Doubling Down",
            summary: "Double your bet for exactly one more card.",
            body: """
            **Double down**: you double your original bet and receive **exactly one** more card, then you must stand.

            Double when you have a strong starting hand and the dealer is weak:
            • **Hard 11** — double against almost everything.
            • **Hard 10** — double against dealer 2–9.
            • **Hard 9** — double against dealer 3–6.
            • **Soft hands (A-2 through A-7)** — double against dealer 4/5/6 (the exact range depends on the hand).

            You can only double on your **first two cards**. If a play says "double" but you've already hit, the fallback is usually to hit (or stand for soft 18/19).

            Doubling is how skilled players make money: you put more money in precisely when you have the advantage.
            """
        ),
        Lesson(
            title: "Splitting Pairs",
            summary: "Turn one pair into two separate hands.",
            body: """
            When your first two cards are a **pair**, you may **split** them into two hands, each getting a new second card. You add a second bet equal to the first.

            The non-negotiable rules:
            • **Always split Aces and 8s.**
              – Aces: two hands each starting at 11 is huge.
              – 8s: 16 is the worst hand in the game; two 8s are far better.
            • **Never split 10s** (a 20 is already excellent) **or 5s** (treat 5-5 as a hard 10 and double instead).

            Everything else depends on the dealer's upcard — split the small/medium pairs mainly when the dealer is weak (2–6). **DAS** ("double after split") makes a few extra splits correct, like 4-4 vs 5/6.
            """
        ),
        Lesson(
            title: "Surrender",
            summary: "Give up half your bet on hopeless hands.",
            body: """
            **Late surrender** lets you forfeit your hand and lose only **half** your bet, before the dealer checks for blackjack. Not every table offers it.

            Surrender is correct only on a few truly bad spots:
            • **Hard 16** vs dealer **9, 10, or Ace**.
            • **Hard 15** vs dealer **10** (and vs Ace if the dealer hits soft 17).

            The logic: if you expect to lose **more than half** the time, giving up half a bet beats playing it out. It feels like quitting, but over thousands of hands it saves money.

            Note: 8-8 vs a 10 is still a **split**, not a surrender, under standard rules.
            """
        ),
        Lesson(
            title: "Betting & Bankroll",
            summary: "Bet sizing, units, and not going broke.",
            body: """
            Think in **units**, not dollars. A unit is one base bet. A common guideline is to bring **40–100 units** so normal swings don't wipe you out.

            • **Flat betting** (same bet every hand) is the safest if you're only using basic strategy — it can't beat the house, but it minimizes variance.
            • **Never** chase losses by doubling your bet after a loss (the "Martingale"). Table limits and bad runs make it a fast way to lose everything.
            • Decide a **stop-loss** and a **win goal** before you sit down, and walk when you hit either.

            Basic strategy alone still leaves a small house edge (~0.5% with good rules). The only legal way to flip the edge is **bet spreading based on the count** — see the counting lessons.
            """
        ),
        Lesson(
            title: "Card Counting (Hi-Lo)",
            summary: "Track the deck's richness to know when to bet big.",
            body: """
            Counting doesn't mean memorizing cards — it's tracking a single number that tells you whether the remaining shoe is rich in **10s and Aces** (good for you) or small cards (good for the dealer).

            **Hi-Lo tags:**
            • Cards **2, 3, 4, 5, 6** → **+1**
            • Cards **7, 8, 9** → **0**
            • Cards **10, J, Q, K, A** → **−1**

            **Running count:** start at 0 and add the tag of every card you see. A high positive count means many small cards are gone, so the remaining deck is rich in tens and Aces — more blackjacks and stronger doubles for you.

            **True count:** divide the running count by the **decks remaining**. A +6 running count with 6 decks left is only a true +1; with 1 deck left it's a powerful +6. The true count is what you actually bet and deviate on.

            **Putting it to work:** bet the minimum at true count ≤ +1, and raise your bet as the true count climbs. When the true count is high, you have the edge — that's the whole game.

            Use the **Counting** tab to drill the tags, keep a running count through a shoe, and convert to a true count under time pressure.
            """
        )
    ]
}
