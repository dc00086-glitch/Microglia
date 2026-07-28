import FamilyControls
import Foundation

/// Shared storage for the user's blocked-app selection, so the main app and the
/// extensions (which run in separate processes) both see the same choice.
///
/// This relies on an **App Group** — add the same App Group ID to the main app
/// target AND each extension target in Xcode, then put it in `appGroupID` below.
enum SelectionStore {
    /// Replace with your real App Group, e.g. "group.com.yourname.movetounlock".
    static let appGroupID = "group.com.example.movetounlock"
    private static let key = "MoveToUnlock.selection"

    private static var defaults: UserDefaults {
        UserDefaults(suiteName: appGroupID) ?? .standard
    }

    static func save(_ selection: FamilyActivitySelection) {
        if let data = try? JSONEncoder().encode(selection) {
            defaults.set(data, forKey: key)
        }
    }

    static func load() -> FamilyActivitySelection {
        guard let data = defaults.data(forKey: key),
              let selection = try? JSONDecoder().decode(FamilyActivitySelection.self, from: data)
        else { return FamilyActivitySelection() }
        return selection
    }
}
