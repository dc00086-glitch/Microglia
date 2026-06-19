# Blackjack Trainer (iPhone)

A native iPhone app for **learning blackjack strategy**: the rules, the full
basic‑strategy chart ("dealer shows X, I have Y → do Z and here's why"), a
decision **trainer**, and **Hi‑Lo card‑counting** practice. 100% offline, no
accounts, no ads, no real‑money gambling.

> For learning only. Gamble responsibly.

## What's inside

| Tab | What it does |
|-----|--------------|
| **Learn** | Short lessons: hand values, hit/stand, doubling, splitting, surrender, betting & bankroll, and card counting. |
| **Strategy** | The interactive basic‑strategy chart (hard totals, soft totals, pairs). Tap any cell to see the play **and the reasoning**. Colors follow the rule set. |
| **Trainer** | Deals random hands and asks for the correct play. Grades you instantly with an explanation; tracks accuracy and streaks. |
| **Counting** | Two Hi‑Lo drills: tag each card (+1/0/−1), and keep a **running count** through a sequence then convert to a **true count**. |
| **Settings** | Table rules (decks, H17/S17, DAS, surrender) that change correct strategy, plus your stats. |

The strategy engine encodes the standard 4–8 deck charts with H17/S17 and
double‑after‑split variations.

## How to run it on your iPhone

You need a **Mac with Xcode** (free from the Mac App Store). You do **not** need
a paid Apple Developer account to run it on your own phone.

1. Copy the `BlackjackTrainer` folder to your Mac.
2. Double‑click **`BlackjackTrainer.xcodeproj`** to open it in Xcode.
3. In the toolbar, pick your iPhone (plug it in) or a Simulator as the run target.
4. For a real device: select the **BlackjackTrainer** target → **Signing &
   Capabilities** → check *Automatically manage signing* and choose your Apple
   ID as the *Team*. (Change the Bundle Identifier to something unique like
   `com.yourname.BlackjackTrainer` if Xcode complains.)
5. Press **▶ Run** (⌘R). The app builds and installs.
6. On the phone, the first launch may need: Settings → General → VPN & Device
   Management → trust your developer certificate.

Free Apple‑ID signing means the app expires after ~7 days; just re‑run from
Xcode to refresh it.

### Alternative: regenerate the project with XcodeGen

If the bundled `.xcodeproj` ever gives you trouble, you can regenerate it:

```bash
brew install xcodegen
cd BlackjackTrainer
xcodegen generate
```

(`project.yml` is included for this.)

## Project layout

```
BlackjackTrainer/
  BlackjackTrainerApp.swift     app entry
  Models/                       Card, Hand, RuleSet, BasicStrategy, Shoe
  State/                        AppSettings, TrainerStats (persisted)
  Views/                        Learn, Strategy chart, Trainer, Counting, Settings
  Resources/Lessons.swift       lesson text
  Assets.xcassets               app icon + accent color
```

Minimum iOS 16. SwiftUI only, no third‑party dependencies.
