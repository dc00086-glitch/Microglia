import Foundation
import Vision

/// A body-weight exercise we can verify from the camera.
///
/// Each case describes a rep as the angle at one joint swinging between an
/// "extended" position and a "flexed" position and back. For a squat that's the
/// knee angle (hip → knee → ankle); for a push-up the elbow angle
/// (shoulder → elbow → wrist); for a sit-up the torso/hip angle
/// (shoulder → hip → knee).
enum Exercise: String, CaseIterable, Identifiable {
    case squat
    case pushup
    case situp

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .squat: return "Squats"
        case .pushup: return "Push-ups"
        case .situp: return "Sit-ups"
        }
    }

    /// Right-side joints whose angle defines a rep, in order A–B–C (angle at B).
    var rightJoints: (a: VNHumanBodyPoseObservation.JointName,
                      b: VNHumanBodyPoseObservation.JointName,
                      c: VNHumanBodyPoseObservation.JointName) {
        switch self {
        case .squat:  return (.rightHip, .rightKnee, .rightAnkle)
        case .pushup: return (.rightShoulder, .rightElbow, .rightWrist)
        case .situp:  return (.rightShoulder, .rightHip, .rightKnee)
        }
    }

    /// Left-side mirror of `rightJoints`. When both sides are visible we average
    /// them, which makes counting steadier and harder to fake with a half-rep.
    var leftJoints: (a: VNHumanBodyPoseObservation.JointName,
                     b: VNHumanBodyPoseObservation.JointName,
                     c: VNHumanBodyPoseObservation.JointName) {
        switch self {
        case .squat:  return (.leftHip, .leftKnee, .leftAnkle)
        case .pushup: return (.leftShoulder, .leftElbow, .leftWrist)
        case .situp:  return (.leftShoulder, .leftHip, .leftKnee)
        }
    }

    /// Below this angle (degrees) the joint counts as "flexed" (bottom of a rep).
    var flexedAngle: Double {
        switch self {
        case .squat: return 100
        case .pushup: return 95
        case .situp: return 95
        }
    }

    /// Above this angle (degrees) the joint counts as "extended" (top of a rep).
    /// A rep is counted on a flexed → extended transition.
    var extendedAngle: Double {
        switch self {
        case .squat: return 160
        case .pushup: return 150
        case .situp: return 150
        }
    }

    /// Minimum seconds a rep must take (flex → extend). Rejects impossibly fast
    /// twitches that aren't real reps, which is the easiest way to cheat.
    var minRepSeconds: Double {
        switch self {
        case .squat: return 0.8
        case .pushup: return 0.7
        case .situp: return 0.7
        }
    }
}
