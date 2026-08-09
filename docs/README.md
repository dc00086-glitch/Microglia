# Blackjack Trainer — Web App (works on iPhone)

A phone-friendly web app for **learning blackjack strategy** and practicing
**Hi-Lo card counting**. No Mac, no App Store, no install needed — it runs in
Safari and can be added to your home screen so it looks and feels like a real
app and works **offline**.

> For learning only. Gamble responsibly.

## Features

- **Learn** — lessons: hand values, hit/stand, doubling, splitting, surrender,
  betting & bankroll, and card counting.
- **Strategy** — the interactive basic-strategy chart. Tap any cell to see the
  correct play **and the reasoning** ("dealer shows X, you have Y → do Z").
- **Trainer** — random hands; pick the play and get graded instantly, with
  accuracy and streak tracking.
- **Count** — Hi-Lo drills: tag each card (+1/0/−1), and keep a running count
  through a sequence then convert to a true count.
- **Settings** — table rules (decks, H17/S17, DAS, surrender) that change the
  correct strategy.

Everything is saved on your phone (local storage). 100% offline after first load.

## Get it on your iPhone (easiest path: GitHub Pages)

**Step 1 — Publish it (one time, ~1 minute):**
1. On GitHub, open this repository → **Settings** → **Pages** (left sidebar).
2. Under *Build and deployment* → *Source*, choose **Deploy from a branch**.
3. Set **Branch** to `claude/iphone-blackjack-app-6uhru7` (or `main` after you
   merge) and **Folder** to **`/docs`**, then **Save**.
4. Wait ~1 minute. GitHub shows a URL like
   `https://dc00086-glitch.github.io/Microglia/`.

**Step 2 — Add to your home screen:**
1. Open that URL in **Safari** on your iPhone.
2. Tap the **Share** button (the square with an up-arrow).
3. Tap **Add to Home Screen** → **Add**.
4. Launch it from the new "Blackjack" icon — it opens full-screen like an app
   and works without internet.

### Try it instantly on a computer

Open a terminal in this `docs` folder and run:

```bash
python3 -m http.server 8000
```

Then visit `http://localhost:8000`. (Open it directly as a `file://` works too,
but the offline service worker only activates when served over http.)

## Files

```
docs/
  index.html              app shell + iOS home-screen meta tags
  styles.css              styling (felt-green theme)
  app.js                  strategy engine, lessons, trainer, counting, UI
  manifest.webmanifest    PWA manifest
  sw.js                   service worker (offline cache)
  icon-180.png / 512.png  home-screen icons
```

No build step, no dependencies — plain HTML/CSS/JS.
