# Spin & Sweat — camera rep counter

A web-based workout roulette. **Spin the wheel** to get a random exercise and rep
count, then the **front camera counts your reps** using in-browser pose detection
(TensorFlow.js MoveNet). Everything runs on your phone — nothing is recorded or
uploaded. No Mac, no App Store, no cost.

## How it works
- Spin the wheel → it lands on something like "15 Squats" or "10 Push-ups".
- Tap **Let's go**, prop your phone so your whole body is in frame.
- It tracks your joints and counts each clean rep (squats = knee angle, push-ups
  = elbow angle, sit-ups = hip angle). Reps that are too fast to be real don't
  count.
- Hit the goal → celebrate → spin again.

## Get it onto your phone (free, ~2 min, no Mac)
The camera only works over **HTTPS**, so it needs hosting. GitHub Pages is free:

1. On GitHub: this repo → **Settings** → **Pages**.
2. **Source** → **Deploy from a branch**.
3. Pick this branch (or merge to your default branch first) and folder **/docs**,
   then **Save**.
4. Wait ~1 minute for the URL, e.g. `https://dc00086-glitch.github.io/Microglia/`.
5. Open it in **Safari on iPhone**, tap **SPIN**, then **Let's go**, and allow
   camera access.

Tip: Safari **Share → Add to Home Screen** makes it launch like an app.

## Tweaking
- **Wheel options:** edit the `SEG` array in `index.html` to change the exercise
  and rep combinations.
- **Counting sensitivity:** adjust the `flexed` / `extended` angle thresholds in
  the `EXERCISES` object if reps aren't registering for your camera setup.
