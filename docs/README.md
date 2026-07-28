# Move to Unlock — Web version

A no-Mac, no-App-Store version of the exercise trainer. It opens in your phone's
browser, uses the front camera, and counts real squats / push-ups / sit-ups with
in-browser pose detection (TensorFlow.js MoveNet). Everything runs on your phone —
nothing is recorded or uploaded.

## What it can and can't do

- ✅ **Counts your reps on camera** — the fun part works fully as a web page.
- ❌ **Cannot block Instagram/TikTok.** A web page is sandboxed and physically
  can't touch other apps. Only the native app (see `../MoveToUnlock/`) can lock
  apps via Apple's Screen Time API.

So use this as your "prove you did the reps" trainer, and pair it with **Screen
Time → App Limits** (Settings app) for the locking — ideally with a passcode a
friend holds so you can't just dismiss the limit.

## Get it onto your phone (free, ~2 minutes, no Mac)

The camera only works over **HTTPS**, so it needs to be hosted. GitHub Pages does
this free:

1. On GitHub, open this repo → **Settings** → **Pages**.
2. Under **Build and deployment → Source**, pick **Deploy from a branch**.
3. Choose the branch `claude/iphone-social-media-exercise-lock-n1ds0k` (or merge
   to your default branch first) and folder **/docs**, then **Save**.
4. Wait ~1 minute. Pages shows a URL like
   `https://dc00086-glitch.github.io/Microglia/`.
5. Open that URL in **Safari on your iPhone**, tap **Start camera**, and allow
   camera access when asked.

Tip: in Safari tap **Share → Add to Home Screen** so it launches like an app.

## Using it

Prop your phone up so your whole body is visible, pick the exercise at the top,
set your rep goal, and go. Green skeleton = it sees you; the big number counts up
each clean rep. It rejects reps that are too fast to be real.

If counting feels off for your setup, tweak the `flexed` / `extended` angle
thresholds in `index.html` (the `EXERCISES` object).
