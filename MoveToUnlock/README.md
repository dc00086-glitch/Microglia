# Move to Unlock

An iPhone app that **blocks your chosen social-media apps until you do a set
number of exercises**, verified live on the front camera (squats, push-ups, or
sit-ups). Built with SwiftUI + Apple's Vision and Screen Time frameworks — no
in-app purchases, no subscription, because you're building it for yourself.

This folder contains the **Swift source files** and a setup guide. It is a
starting scaffold, not a pre-built `.xcodeproj` — iOS apps have to be assembled
in Xcode on a Mac, so the steps below wire these files into a project.

---

## How it works

1. **Pick apps to block.** `FamilyActivityPicker` lets you choose Instagram,
   TikTok, etc. (or whole categories like "Social").
2. **Lock.** `ShieldManager` uses `ManagedSettingsStore` to shield them — tapping
   a locked app shows a block screen instead of opening it.
3. **Earn an unlock.** Point the front camera at yourself and do the reps.
   `CameraPoseModel` runs Vision's `VNDetectHumanBodyPoseRequest` on every frame
   to find your body joints; `RepCounter` watches one joint's angle (e.g. the
   knee for squats) swing down and back up to count a clean rep.
4. **Unlock.** When you hit the goal, the shield lifts for a reward window
   (15 min by default), then re-locks.

### Files
| File | Role |
|------|------|
| `MoveToUnlockApp.swift` | App entry point, requests Screen Time permission |
| `Views/ContentView.swift` | Home: pick apps, exercise, and rep goal |
| `Views/WorkoutView.swift` | Camera screen with skeleton overlay + rep counter |
| `Views/CameraPreview.swift` | SwiftUI wrapper around the camera preview layer |
| `Vision/CameraPoseModel.swift` | Camera capture + pose detection + rep detection |
| `Models/Exercise.swift` | The three exercises and their joint/angle rules |
| `Models/RepCounter.swift` | Rep state machine + joint-angle math |
| `ScreenTime/ShieldManager.swift` | Applies/lifts the app block |
| `ScreenTime/SelectionStore.swift` | Shares the blocked-app list with the extensions (App Group) |
| `Extensions/DeviceActivityMonitorExtension.swift` | Re-locks on a schedule so the block survives a force-quit |
| `Extensions/ShieldConfigurationExtension.swift` | Custom "do your reps" block screen |

---

## Setup (what you need)

- A **Mac** with **Xcode 15+**.
- An **iPhone** running iOS 16+ (the camera and Screen Time APIs don't work in
  the Simulator — pose detection needs a real camera).
- A **free Apple ID** is enough to run it on your own phone for 7 days at a time.
  A paid **Apple Developer account** ($99/yr) makes it permanent and is
  recommended because the Screen Time entitlement is smoother to get.

### Steps
1. **New project** in Xcode → iOS → App → SwiftUI. Name it `MoveToUnlock`.
2. **Add these files**: drag the `Models`, `Vision`, `ScreenTime`, and `Views`
   folders plus `MoveToUnlockApp.swift` into the project (replace the default
   `ContentView.swift` and `App.swift`).
3. **Add capabilities** (Signing & Capabilities tab):
   - **Family Controls** — add the capability. This adds the
     `com.apple.developer.family-controls` entitlement.
   - You may need to **request approval** from Apple for Family Controls at
     https://developer.apple.com/contact/request/family-controls-distribution
     (approval is required for App Store release; for personal on-device use a
     development build generally works once the capability is added).
4. **Add Info.plist keys** (Info tab → add rows):
   - `NSCameraUsageDescription` → e.g. "Used to count your exercise reps."
5. **Import frameworks** are already in the source (`FamilyControls`,
   `ManagedSettings`, `Vision`, `AVFoundation`) — no manual linking needed.
6. **Run on your iPhone** (not the Simulator). Grant Screen Time and Camera
   permission when prompted.

---

## Adding the two extensions (makes the lock stick)

The files in `Extensions/` are already written — you just create the targets and
drop them in:

1. **App Group** (shared storage both processes read):
   - Main app target → Signing & Capabilities → **+ App Groups** → add
     e.g. `group.com.yourname.movetounlock`.
   - Put that exact string in `SelectionStore.appGroupID`.
2. **Device Activity Monitor extension:** File → New → Target →
   **Device Activity Monitor Extension**. Delete its stub file, add
   `DeviceActivityMonitorExtension.swift` and `SelectionStore.swift` to that
   target, and give the target the same App Group.
3. **Shield Configuration extension:** File → New → Target →
   **Shield Configuration Extension**. Same idea — add
   `ShieldConfigurationExtension.swift` to it.

With these in place the block re-asserts itself even if the app is killed, and
tapping a locked app shows your custom "do your reps" screen.

---

## Getting it onto your iPhone (the honest truth)

A **custom camera/motion app cannot be built entirely on an iPhone** — Apple
requires its build tools (Xcode), which only run on a Mac. There is no on-phone
shortcut for *this* kind of app. Your realistic paths, easiest first:

1. **Any Mac for ~30 minutes** — a WVU campus computer lab, a friend's laptop.
   Plug your iPhone in with a cable, press Run, done. A **free Apple ID** installs
   it on your own phone (re-run every 7 days); the $99/yr account makes it
   permanent.
2. **Rent a cloud Mac** (MacinCloud ~$1/hr) — build an installable file there.
3. **Build with GitHub Actions (no Mac at all)** — its free macOS runners can
   compile the app into an `.ipa`, which you then sideload with **AltStore** or
   **Sideloadly** using your Apple ID. Zero Mac, but the signing setup is a
   project in itself.

An **iPad** with Apple's free **Swift Playgrounds** app can build/run SwiftUI on
the iPad itself, but the Screen Time entitlement + extensions here are hard to
set up there — good for prototyping the camera half only.

## Honest limitations

- **iOS won't let any app truly force-block another.** Screen Time shields are
  the strongest tool available, but a determined user can open Settings and turn
  Screen Time off, or delete this app. Self-control apps all share this ceiling —
  the value is friction, not a hard lock.
- **Pose rep-counting isn't perfect.** Vision is good but lighting, camera angle,
  and partial views (a push-up where your legs are off-screen) affect accuracy.
  The angle thresholds in `Exercise.swift` are starting values — tune
  `flexedAngle` / `extendedAngle` per exercise once you test on yourself.
- **You can "cheat" the camera** by doing sloppy reps. You could tighten this by
  requiring both left and right joints to agree, checking torso orientation, or
  enforcing a minimum time per rep.
