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

## Making it more robust (production notes)

The scaffold keeps everything inside the main app to stay readable. Two upgrades
matter if you want it to hold up:

- **Re-lock reliably.** Right now the unlock timer runs in-app; if you force-quit
  the app the re-lock won't fire. The proper fix is a **DeviceActivityMonitor
  app extension** that re-applies the shield on a schedule, sharing the app
  selection through an **App Group**. Apple's sample "Screen Time API" project
  shows the pattern.
- **Custom block screen.** A **ShieldConfiguration extension** lets you replace
  the default block screen with your own ("Do 10 squats to unlock 💪").

---

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
