import CoreGraphics
import Foundation

/// Counts repetitions for one exercise by running a small state machine over a
/// stream of joint angles.
///
/// The rep is only counted when the joint goes flexed (bottom) and then returns
/// to extended (top). Requiring both halves stops half-reps and jitter from
/// inflating the count.
final class RepCounter {
    private enum Phase { case extended, flexed }

    private let exercise: Exercise
    private var phase: Phase = .extended
    private(set) var reps: Int = 0

    init(exercise: Exercise) {
        self.exercise = exercise
    }

    func reset() {
        phase = .extended
        reps = 0
    }

    /// Feed one frame's angle (in degrees) at the tracked joint.
    /// Returns true if this update completed a rep.
    @discardableResult
    func update(angle: Double) -> Bool {
        switch phase {
        case .extended:
            if angle < exercise.flexedAngle {
                phase = .flexed
            }
        case .flexed:
            if angle > exercise.extendedAngle {
                phase = .extended
                reps += 1
                return true
            }
        }
        return false
    }
}

/// Angle in degrees at point `b`, formed by the segments b→a and b→c.
/// Returns nil if the points coincide (undefined angle).
func angleBetween(_ a: CGPoint, _ b: CGPoint, _ c: CGPoint) -> Double? {
    let v1 = CGVector(dx: a.x - b.x, dy: a.y - b.y)
    let v2 = CGVector(dx: c.x - b.x, dy: c.y - b.y)
    let mag1 = (v1.dx * v1.dx + v1.dy * v1.dy).squareRoot()
    let mag2 = (v2.dx * v2.dx + v2.dy * v2.dy).squareRoot()
    guard mag1 > 0, mag2 > 0 else { return nil }
    let dot = v1.dx * v2.dx + v1.dy * v2.dy
    let cosine = max(-1, min(1, dot / (mag1 * mag2)))
    return acos(cosine) * 180 / .pi
}
