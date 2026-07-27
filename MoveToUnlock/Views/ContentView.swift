import FamilyControls
import SwiftUI

/// Home screen: pick which apps to block, choose an exercise + goal, and start.
struct ContentView: View {
    @EnvironmentObject private var shield: ShieldManager

    @State private var exercise: Exercise = .squat
    @State private var goal = 10
    @State private var showingPicker = false
    @State private var showingWorkout = false

    var body: some View {
        NavigationStack {
            Form {
                Section("What to lock") {
                    Button {
                        showingPicker = true
                    } label: {
                        Label(selectionSummary, systemImage: "lock.app.dashed")
                    }
                    if !shield.isAuthorized {
                        Text("Screen Time permission not granted yet.")
                            .font(.footnote)
                            .foregroundStyle(.orange)
                    }
                }

                Section("Unlock requirement") {
                    Picker("Exercise", selection: $exercise) {
                        ForEach(Exercise.allCases) { ex in
                            Text(ex.displayName).tag(ex)
                        }
                    }
                    Stepper("\(goal) reps", value: $goal, in: 1...100)
                }

                Section {
                    Button {
                        shield.lock()
                    } label: {
                        Label("Lock now", systemImage: "lock.fill")
                    }
                    .disabled(shield.selection.applicationTokens.isEmpty &&
                              shield.selection.categoryTokens.isEmpty)

                    Button {
                        showingWorkout = true
                    } label: {
                        Label("Do \(goal) \(exercise.displayName.lowercased()) to unlock",
                              systemImage: "figure.strengthtraining.functional")
                    }
                    .disabled(!shield.isLocked)
                }

                if shield.isLocked {
                    Section {
                        Label("Apps are locked", systemImage: "lock.fill")
                            .foregroundStyle(.red)
                    }
                }
            }
            .navigationTitle("Move to Unlock")
            .familyActivityPicker(isPresented: $showingPicker,
                                  selection: $shield.selection)
            .fullScreenCover(isPresented: $showingWorkout) {
                WorkoutView(exercise: exercise, goal: goal) {
                    shield.unlock(for: 15 * 60) // 15-minute reward window
                    showingWorkout = false
                }
            }
        }
    }

    private var selectionSummary: String {
        let apps = shield.selection.applicationTokens.count
        let cats = shield.selection.categoryTokens.count
        if apps == 0 && cats == 0 { return "Choose apps to block" }
        var parts: [String] = []
        if apps > 0 { parts.append("\(apps) app\(apps == 1 ? "" : "s")") }
        if cats > 0 { parts.append("\(cats) categor\(cats == 1 ? "y" : "ies")") }
        return parts.joined(separator: ", ")
    }
}
