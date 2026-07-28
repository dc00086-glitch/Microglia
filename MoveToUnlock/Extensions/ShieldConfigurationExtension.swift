import ManagedSettings
import ManagedSettingsUI
import UIKit

/// Customises the block screen the user sees when they tap a shielded app.
///
/// This lives in a **Shield Configuration app extension** target (Xcode:
/// File → New → Target → Shield Configuration Extension). It runs in its own
/// process, so keep it self-contained.
final class ShieldConfigurationExtension: ShieldConfigurationDataSource {
    private func config() -> ShieldConfiguration {
        ShieldConfiguration(
            backgroundBlurStyle: .systemMaterialDark,
            backgroundColor: nil,
            icon: UIImage(systemName: "figure.strengthtraining.functional"),
            title: .init(text: "Locked", color: .white),
            subtitle: .init(text: "Open Move to Unlock and do your reps to earn this back.",
                            color: .white.withAlphaComponent(0.8)),
            primaryButtonLabel: .init(text: "Got it", color: .black),
            primaryButtonBackgroundColor: .white
        )
    }

    override func configuration(shielding application: Application) -> ShieldConfiguration {
        config()
    }

    override func configuration(shielding application: Application,
                                in category: ActivityCategory) -> ShieldConfiguration {
        config()
    }

    override func configuration(shielding webDomain: WebDomain) -> ShieldConfiguration {
        config()
    }

    override func configuration(shielding webDomain: WebDomain,
                                in category: ActivityCategory) -> ShieldConfiguration {
        config()
    }
}
