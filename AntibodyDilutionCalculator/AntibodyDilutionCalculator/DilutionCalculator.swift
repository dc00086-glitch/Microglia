import Foundation

/// Units for volume input/output
enum VolumeUnit: String, CaseIterable, Identifiable {
    case uL = "µL"
    case mL = "mL"
    case L = "L"

    var id: String { rawValue }

    /// Convert a value in this unit to µL
    func toMicroliters(_ value: Double) -> Double {
        switch self {
        case .uL: return value
        case .mL: return value * 1_000
        case .L:  return value * 1_000_000
        }
    }

    /// Convert a value in µL to this unit
    func fromMicroliters(_ value: Double) -> Double {
        switch self {
        case .uL: return value
        case .mL: return value / 1_000
        case .L:  return value / 1_000_000
        }
    }
}

/// Results from a dilution calculation
struct DilutionResult {
    let antibodyVolume: Double   // in µL
    let diluentVolume: Double    // in µL
    let totalVolume: Double      // in µL
    let dilutionRatio: Double    // e.g. 1000 means 1:1000
    let numberOfSamples: Int
    let perSampleTotal: Double   // in µL
    let perSampleAntibody: Double // in µL
    let perSampleDiluent: Double  // in µL

    /// Format a volume in µL to the best human-readable unit
    static func formatVolume(_ microliters: Double) -> String {
        if microliters >= 1_000_000 {
            return String(format: "%.4g L", microliters / 1_000_000)
        } else if microliters >= 1_000 {
            return String(format: "%.4g mL", microliters / 1_000)
        } else {
            return String(format: "%.4g µL", microliters)
        }
    }
}

struct DilutionCalculator {

    // MARK: - Mode 1: Given total volume + dilution ratio

    /// Calculate antibody and diluent volumes from total volume and dilution ratio.
    /// - Parameters:
    ///   - totalVolume: desired total volume in µL (per sample)
    ///   - dilutionRatio: the "X" in 1:X  (e.g. 1000 for 1:1000)
    ///   - numberOfSamples: how many samples to prepare
    static func fromTotalAndRatio(
        totalVolume: Double,
        dilutionRatio: Double,
        numberOfSamples: Int = 1
    ) -> DilutionResult? {
        guard totalVolume > 0, dilutionRatio > 0, numberOfSamples > 0 else { return nil }

        let perSampleAntibody = totalVolume / dilutionRatio
        let perSampleDiluent = totalVolume - perSampleAntibody

        return DilutionResult(
            antibodyVolume: perSampleAntibody * Double(numberOfSamples),
            diluentVolume: perSampleDiluent * Double(numberOfSamples),
            totalVolume: totalVolume * Double(numberOfSamples),
            dilutionRatio: dilutionRatio,
            numberOfSamples: numberOfSamples,
            perSampleTotal: totalVolume,
            perSampleAntibody: perSampleAntibody,
            perSampleDiluent: perSampleDiluent
        )
    }

    // MARK: - Mode 2: Given antibody volume + dilution ratio

    /// Calculate total volume and diluent from antibody volume and dilution ratio.
    static func fromAntibodyAndRatio(
        antibodyVolume: Double,
        dilutionRatio: Double,
        numberOfSamples: Int = 1
    ) -> DilutionResult? {
        guard antibodyVolume > 0, dilutionRatio > 0, numberOfSamples > 0 else { return nil }

        let perSampleAntibody = antibodyVolume
        let perSampleTotal = perSampleAntibody * dilutionRatio
        let perSampleDiluent = perSampleTotal - perSampleAntibody

        return DilutionResult(
            antibodyVolume: perSampleAntibody * Double(numberOfSamples),
            diluentVolume: perSampleDiluent * Double(numberOfSamples),
            totalVolume: perSampleTotal * Double(numberOfSamples),
            dilutionRatio: dilutionRatio,
            numberOfSamples: numberOfSamples,
            perSampleTotal: perSampleTotal,
            perSampleAntibody: perSampleAntibody,
            perSampleDiluent: perSampleDiluent
        )
    }

    // MARK: - Mode 3: Given total volume + antibody volume

    /// Calculate the dilution ratio from total volume and antibody volume.
    static func fromTotalAndAntibody(
        totalVolume: Double,
        antibodyVolume: Double,
        numberOfSamples: Int = 1
    ) -> DilutionResult? {
        guard totalVolume > 0, antibodyVolume > 0, antibodyVolume < totalVolume, numberOfSamples > 0 else { return nil }

        let ratio = totalVolume / antibodyVolume
        let perSampleDiluent = totalVolume - antibodyVolume

        return DilutionResult(
            antibodyVolume: antibodyVolume * Double(numberOfSamples),
            diluentVolume: perSampleDiluent * Double(numberOfSamples),
            totalVolume: totalVolume * Double(numberOfSamples),
            dilutionRatio: ratio,
            numberOfSamples: numberOfSamples,
            perSampleTotal: totalVolume,
            perSampleAntibody: antibodyVolume,
            perSampleDiluent: perSampleDiluent
        )
    }
}
