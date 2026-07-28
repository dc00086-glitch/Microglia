import DeviceActivity
import ManagedSettings

/// Re-applies the shield on a schedule so the block survives even if the main
/// app is force-quit. This is what makes the lock actually stick.
///
/// This lives in a **Device Activity Monitor app extension** target (Xcode:
/// File → New → Target → Device Activity Monitor Extension). The main app
/// starts monitoring with a schedule; iOS then calls these callbacks in this
/// separate process at the scheduled times.
final class DeviceActivityMonitorExtension: DeviceActivityMonitor {
    private let store = ManagedSettingsStore()

    /// Called at the start of each monitored interval — re-assert the shield.
    override func intervalDidStart(for activity: DeviceActivityName) {
        super.intervalDidStart(for: activity)
        applyShield()
    }

    /// Called at the end of an interval — keep it shielded (the reward window is
    /// handled by the app clearing the shield, not by this callback).
    override func intervalDidEnd(for activity: DeviceActivityName) {
        super.intervalDidEnd(for: activity)
        applyShield()
    }

    private func applyShield() {
        let selection = SelectionStore.load()
        let apps = selection.applicationTokens
        let categories = selection.categoryTokens
        store.shield.applications = apps.isEmpty ? nil : apps
        store.shield.applicationCategories = categories.isEmpty ? nil : .specific(categories)
    }
}
