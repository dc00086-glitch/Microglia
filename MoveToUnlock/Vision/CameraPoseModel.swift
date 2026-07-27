import AVFoundation
import Combine
import CoreGraphics
import Vision

/// Owns the camera session, runs Vision body-pose detection on every frame, and
/// publishes the current joint positions and rep count to SwiftUI.
///
/// This is the heart of the "prove it on camera" feature. AVFoundation delivers
/// frames; Vision's `VNDetectHumanBodyPoseRequest` turns each frame into ~19
/// body joints; `RepCounter` turns the tracked joint's angle into reps.
@MainActor
final class CameraPoseModel: NSObject, ObservableObject {
    /// Normalized (0...1) joint positions for drawing the skeleton overlay.
    @Published var points: [VNHumanBodyPoseObservation.JointName: CGPoint] = [:]
    @Published var reps: Int = 0
    @Published var currentAngle: Double?
    @Published var isRunning = false

    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "camera.session")
    private let visionQueue = DispatchQueue(label: "vision.work")

    private var exercise: Exercise = .squat
    private var counter = RepCounter(exercise: .squat)
    private let minConfidence: Float = 0.3

    /// Called (on the main actor) when the target rep count is reached.
    var onGoalReached: (() -> Void)?
    private var goal: Int = 0

    func configure() {
        sessionQueue.async { [weak self] in
            self?.setUpSession()
        }
    }

    func start(exercise: Exercise, goal: Int) {
        self.exercise = exercise
        self.goal = goal
        self.counter = RepCounter(exercise: exercise)
        self.reps = 0
        sessionQueue.async { [weak self] in
            guard let self, !self.session.isRunning else { return }
            self.session.startRunning()
            Task { @MainActor in self.isRunning = true }
        }
    }

    func stop() {
        sessionQueue.async { [weak self] in
            guard let self, self.session.isRunning else { return }
            self.session.stopRunning()
            Task { @MainActor in self.isRunning = false }
        }
    }

    private func setUpSession() {
        session.beginConfiguration()
        session.sessionPreset = .high

        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                   for: .video,
                                                   position: .front),
              let input = try? AVCaptureDeviceInput(device: camera),
              session.canAddInput(input) else {
            session.commitConfiguration()
            return
        }
        session.addInput(input)

        videoOutput.setSampleBufferDelegate(self, queue: visionQueue)
        videoOutput.alwaysDiscardsLateVideoFrames = true
        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
        }
        session.commitConfiguration()
    }

    private func process(observation: VNHumanBodyPoseObservation) {
        guard let recognized = try? observation.recognizedPoints(.all) else { return }

        // Collect confident joints as normalized points (Vision origin is
        // bottom-left; flip Y so it matches UIKit's top-left for drawing).
        var mapped: [VNHumanBodyPoseObservation.JointName: CGPoint] = [:]
        for (name, point) in recognized where point.confidence > minConfidence {
            mapped[name] = CGPoint(x: point.location.x, y: 1 - point.location.y)
        }

        let joints = exercise.joints
        var angle: Double?
        if let a = mapped[joints.a], let b = mapped[joints.b], let c = mapped[joints.c] {
            angle = angleBetween(a, b, c)
        }

        let didRep = angle.map { counter.update(angle: $0) } ?? false

        Task { @MainActor in
            self.points = mapped
            self.currentAngle = angle
            if didRep {
                self.reps = self.counter.reps
                if self.counter.reps >= self.goal {
                    self.onGoalReached?()
                }
            }
        }
    }
}

extension CameraPoseModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(_ output: AVCaptureOutput,
                                   didOutput sampleBuffer: CMSampleBuffer,
                                   from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let request = VNDetectHumanBodyPoseRequest()
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                            orientation: .leftMirrored,
                                            options: [:])
        try? handler.perform([request])
        guard let observation = request.results?.first else { return }
        Task { @MainActor in self.process(observation: observation) }
    }
}
