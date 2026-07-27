import SwiftUI
import Vision

/// Full-screen camera workout: shows the live feed with a skeleton overlay and
/// the rep counter, and calls `onComplete` when the goal is hit.
struct WorkoutView: View {
    let exercise: Exercise
    let goal: Int
    let onComplete: () -> Void

    @StateObject private var camera = CameraPoseModel()
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ZStack {
            CameraPreview(session: camera.session)
                .ignoresSafeArea()

            PoseOverlay(points: camera.points)
                .ignoresSafeArea()

            VStack {
                HStack {
                    Button {
                        camera.stop()
                        dismiss()
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.largeTitle)
                            .foregroundStyle(.white.opacity(0.9))
                    }
                    Spacer()
                }
                .padding()

                Spacer()

                VStack(spacing: 4) {
                    Text("\(camera.reps) / \(goal)")
                        .font(.system(size: 72, weight: .bold, design: .rounded))
                        .foregroundStyle(.white)
                    Text(exercise.displayName)
                        .font(.title3)
                        .foregroundStyle(.white.opacity(0.85))
                    if let angle = camera.currentAngle {
                        Text("\(Int(angle))°")
                            .font(.footnote.monospaced())
                            .foregroundStyle(.white.opacity(0.6))
                    }
                }
                .padding(.bottom, 60)
                .shadow(radius: 6)
            }
        }
        .onAppear {
            camera.onGoalReached = {
                camera.stop()
                onComplete()
            }
            camera.configure()
            camera.start(exercise: exercise, goal: goal)
        }
        .onDisappear { camera.stop() }
    }
}

/// Draws lines between the joints Vision reports, in view coordinates.
private struct PoseOverlay: View {
    let points: [VNHumanBodyPoseObservation.JointName: CGPoint]

    private let bones: [(VNHumanBodyPoseObservation.JointName,
                         VNHumanBodyPoseObservation.JointName)] = [
        (.neck, .rightShoulder), (.rightShoulder, .rightElbow), (.rightElbow, .rightWrist),
        (.neck, .leftShoulder), (.leftShoulder, .leftElbow), (.leftElbow, .leftWrist),
        (.neck, .root),
        (.root, .rightHip), (.rightHip, .rightKnee), (.rightKnee, .rightAnkle),
        (.root, .leftHip), (.leftHip, .leftKnee), (.leftKnee, .leftAnkle)
    ]

    var body: some View {
        GeometryReader { geo in
            Path { path in
                for (a, b) in bones {
                    guard let pa = points[a], let pb = points[b] else { continue }
                    path.move(to: scaled(pa, geo.size))
                    path.addLine(to: scaled(pb, geo.size))
                }
            }
            .stroke(.green.opacity(0.85), lineWidth: 4)
        }
    }

    private func scaled(_ p: CGPoint, _ size: CGSize) -> CGPoint {
        CGPoint(x: p.x * size.width, y: p.y * size.height)
    }
}
