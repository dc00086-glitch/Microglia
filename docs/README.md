# Sweat Slots — camera rep counter

A slot-machine workout. **Pull the lever** on three reels of exercise symbols to
get a random mini-circuit, then the **front camera counts your reps** with
in-browser pose detection (TensorFlow.js MoveNet). Everything runs on your phone —
nothing is recorded or uploaded. No Mac, no App Store, no cost.

## How it works
- Pull the slot → the 3 reels land on exercises (🦵 Squats / 💪 Push-ups / 🔥 Sit-ups).
- You get a **3-exercise circuit** — e.g. 10 Squats, then 5 Push-ups, then 15 Sit-ups.
- **Match all three reels = JACKPOT**: one big 30-rep set of that exercise.
- The camera tracks your joints and counts each clean rep (squats = knee angle,
  push-ups = elbow angle, sit-ups = hip angle). Reps too fast to be real don't count.
- Finish a set → tap Continue for the next → celebrate → pull again.

## Get it onto your phone (free, ~2 min, no Mac)
The camera only works over **HTTPS**, so it needs hosting. GitHub Pages is free:

1. On GitHub: this repo → **Settings** → **Pages**.
2. **Source** → **Deploy from a branch**.
3. Pick this branch (or merge to your default branch first) and folder **/docs**,
   then **Save**.
4. Wait ~1 minute for the URL, e.g. `https://dc00086-glitch.github.io/Microglia/`.
5. Open it in **Safari on iPhone**, tap **PULL**, then **Let's go**, and allow camera access.

Tip: Safari **Share → Add to Home Screen** makes it launch like an app.

## Tweaking (in `index.html`)
- **Rep amounts:** edit `REP_CHOICES` (per set) and `JACKPOT_REPS`.
- **Exercises on the reels:** edit `EX_ORDER` / the `EXERCISES` object.
- **Counting sensitivity:** adjust each exercise's `flexed` / `extended` angles.
