import Combine
import FamilyControls
import Foundation
import ManagedSettings

/// Applies and lifts the block ("shield") on the user's chosen apps using
/// Apple's Screen Time frameworks.
///
/// FamilyControls: asks permission and lets the user pick which apps to block.
/// ManagedSettings: actually shields them — tapping a shielded app shows a
/// blocking screen instead of opening it.
///
/// NOTE: This requires the `com.apple.developer.family-controls` entitlement,
/// which you request from Apple. See README for the full setup.
@MainActor
final class ShieldManager: ObservableObject {
    /// The apps/categories the user chose to lock. Bound to a FamilyActivityPicker.
    @Published var selection = FamilyActivitySelection() {
        didSet { persistSelection() }
    }
    @Published var isAuthorized = false
    @Published var isLocked = false

    private let store = ManagedSettingsStore()
    private let selectionKey = "MoveToUnlock.selection"

    init() {
        loadSelection()
    }

    /// Ask the user to grant Screen Time control. Call once at first launch.
    func requestAuthorization() async {
        do {
            try await AuthorizationCenter.shared.requestAuthorization(for: .individual)
            isAuthorized = true
        } catch {
            isAuthorized = false
        }
    }

    /// Turn the block ON — shield every selected app and category.
    func lock() {
        let apps = selection.applicationTokens
        let categories = selection.categoryTokens
        store.shield.applications = apps.isEmpty ? nil : apps
        store.shield.applicationCategories = categories.isEmpty
            ? nil
            : .specific(categories)
        isLocked = true
    }

    /// Turn the block OFF — clear all shields (call this once exercise is verified).
    func unlock() {
        store.shield.applications = nil
        store.shield.applicationCategories = nil
        isLocked = false
    }

    /// Unlock for a fixed window, then automatically re-lock.
    /// For a scaffold this is a simple in-app timer; for production you'd re-lock
    /// from a DeviceActivityMonitor extension so it survives the app closing.
    func unlock(for seconds: TimeInterval) {
        unlock()
        Task {
            try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
            lock()
        }
    }

    private func persistSelection() {
        if let data = try? JSONEncoder().encode(selection) {
            UserDefaults.standard.set(data, forKey: selectionKey)
        }
    }

    private func loadSelection() {
        guard let data = UserDefaults.standard.data(forKey: selectionKey),
              let saved = try? JSONDecoder().decode(FamilyActivitySelection.self, from: data)
        else { return }
        selection = saved
    }
}
