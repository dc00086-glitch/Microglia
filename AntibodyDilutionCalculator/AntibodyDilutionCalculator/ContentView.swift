import SwiftUI

/// Calculation mode — which two values does the user provide?
enum CalcMode: String, CaseIterable, Identifiable {
    case totalAndRatio    = "Total Volume + Dilution Ratio"
    case antibodyAndRatio = "Antibody Volume + Dilution Ratio"
    case totalAndAntibody = "Total Volume + Antibody Volume"

    var id: String { rawValue }
}

struct ContentView: View {
    @State private var mode: CalcMode = .totalAndRatio

    // Input strings (text fields)
    @State private var totalVolumeText = ""
    @State private var antibodyVolumeText = ""
    @State private var dilutionRatioText = ""
    @State private var numberOfSamplesText = "1"

    // Unit pickers
    @State private var totalVolumeUnit: VolumeUnit = .mL
    @State private var antibodyVolumeUnit: VolumeUnit = .uL

    // Result
    @State private var result: DilutionResult?
    @State private var errorMessage: String?

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Header
                Text("Antibody Dilution Calculator")
                    .font(.title.bold())
                    .padding(.top)

                // Mode picker
                VStack(alignment: .leading, spacing: 6) {
                    Text("I want to calculate from:")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    Picker("Mode", selection: $mode) {
                        ForEach(CalcMode.allCases) { m in
                            Text(m.rawValue).tag(m)
                        }
                    }
                    .pickerStyle(.segmented)
                    .onChange(of: mode) { clearResults() }
                }

                Divider()

                // Input fields — show/hide based on mode
                VStack(spacing: 14) {
                    if mode == .totalAndRatio || mode == .totalAndAntibody {
                        volumeField(
                            label: "Total Volume (per sample)",
                            text: $totalVolumeText,
                            unit: $totalVolumeUnit
                        )
                    }

                    if mode == .antibodyAndRatio || mode == .totalAndAntibody {
                        volumeField(
                            label: "Antibody Volume (per sample)",
                            text: $antibodyVolumeText,
                            unit: $antibodyVolumeUnit
                        )
                    }

                    if mode == .totalAndRatio || mode == .antibodyAndRatio {
                        HStack {
                            Text("Dilution Ratio  1 :")
                                .frame(width: 140, alignment: .leading)
                            TextField("e.g. 1000", text: $dilutionRatioText)
                                #if os(iOS)
                                .keyboardType(.decimalPad)
                                #endif
                                .textFieldStyle(.roundedBorder)
                        }
                    }

                    HStack {
                        Text("Number of Samples")
                            .frame(width: 140, alignment: .leading)
                        TextField("1", text: $numberOfSamplesText)
                            #if os(iOS)
                            .keyboardType(.numberPad)
                            #endif
                            .textFieldStyle(.roundedBorder)
                    }
                }

                // Calculate button
                Button(action: calculate) {
                    Text("Calculate")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                }
                .buttonStyle(.borderedProminent)
                .tint(.blue)

                // Error
                if let errorMessage {
                    Text(errorMessage)
                        .foregroundStyle(.red)
                        .font(.callout)
                }

                // Results card
                if let result {
                    ResultsCard(result: result)
                }

                Spacer(minLength: 20)
            }
            .padding(.horizontal)
        }
        #if os(macOS)
        .frame(maxWidth: 520)
        #endif
    }

    // MARK: - Helpers

    private func volumeField(
        label: String,
        text: Binding<String>,
        unit: Binding<VolumeUnit>
    ) -> some View {
        HStack {
            Text(label)
                .frame(width: 140, alignment: .leading)
                .lineLimit(2)
                .font(.callout)
            TextField("0", text: text)
                #if os(iOS)
                .keyboardType(.decimalPad)
                #endif
                .textFieldStyle(.roundedBorder)
            Picker("", selection: unit) {
                ForEach(VolumeUnit.allCases) { u in
                    Text(u.rawValue).tag(u)
                }
            }
            .frame(width: 70)
        }
    }

    private func clearResults() {
        result = nil
        errorMessage = nil
    }

    private func calculate() {
        errorMessage = nil
        result = nil

        let samples = Int(numberOfSamplesText) ?? 1
        guard samples > 0 else {
            errorMessage = "Number of samples must be at least 1."
            return
        }

        switch mode {
        case .totalAndRatio:
            guard let tv = Double(totalVolumeText), tv > 0 else {
                errorMessage = "Enter a valid total volume."
                return
            }
            guard let dr = Double(dilutionRatioText), dr > 0 else {
                errorMessage = "Enter a valid dilution ratio (the X in 1:X)."
                return
            }
            let tvMicro = totalVolumeUnit.toMicroliters(tv)
            result = DilutionCalculator.fromTotalAndRatio(
                totalVolume: tvMicro, dilutionRatio: dr, numberOfSamples: samples
            )

        case .antibodyAndRatio:
            guard let av = Double(antibodyVolumeText), av > 0 else {
                errorMessage = "Enter a valid antibody volume."
                return
            }
            guard let dr = Double(dilutionRatioText), dr > 0 else {
                errorMessage = "Enter a valid dilution ratio (the X in 1:X)."
                return
            }
            let avMicro = antibodyVolumeUnit.toMicroliters(av)
            result = DilutionCalculator.fromAntibodyAndRatio(
                antibodyVolume: avMicro, dilutionRatio: dr, numberOfSamples: samples
            )

        case .totalAndAntibody:
            guard let tv = Double(totalVolumeText), tv > 0 else {
                errorMessage = "Enter a valid total volume."
                return
            }
            guard let av = Double(antibodyVolumeText), av > 0 else {
                errorMessage = "Enter a valid antibody volume."
                return
            }
            let tvMicro = totalVolumeUnit.toMicroliters(tv)
            let avMicro = antibodyVolumeUnit.toMicroliters(av)
            guard avMicro < tvMicro else {
                errorMessage = "Antibody volume must be less than total volume."
                return
            }
            result = DilutionCalculator.fromTotalAndAntibody(
                totalVolume: tvMicro, antibodyVolume: avMicro, numberOfSamples: samples
            )
        }

        if result == nil && errorMessage == nil {
            errorMessage = "Could not calculate. Check your inputs."
        }
    }
}

// MARK: - Results Card

struct ResultsCard: View {
    let result: DilutionResult

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Results")
                .font(.title3.bold())

            Divider()

            Text("Dilution Ratio")
                .font(.caption)
                .foregroundStyle(.secondary)
            Text("1 : \(String(format: "%.4g", result.dilutionRatio))")
                .font(.title2.bold())
                .foregroundStyle(.blue)

            if result.numberOfSamples > 1 {
                Group {
                    sectionHeader("Per Sample")
                    resultRow("Antibody", result.perSampleAntibody)
                    resultRow("Diluent / Buffer", result.perSampleDiluent)
                    resultRow("Total", result.perSampleTotal)

                    Divider()

                    sectionHeader("All \(result.numberOfSamples) Samples Combined")
                    resultRow("Antibody", result.antibodyVolume)
                    resultRow("Diluent / Buffer", result.diluentVolume)
                    resultRow("Total", result.totalVolume)
                }
            } else {
                resultRow("Antibody Volume", result.antibodyVolume)
                resultRow("Diluent / Buffer Volume", result.diluentVolume)
                resultRow("Total Volume", result.totalVolume)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.secondarySystemBackgroundCompat))
        )
        .padding(.top, 4)
    }

    private func sectionHeader(_ title: String) -> some View {
        Text(title)
            .font(.subheadline.bold())
            .foregroundStyle(.secondary)
            .padding(.top, 4)
    }

    private func resultRow(_ label: String, _ microliters: Double) -> some View {
        HStack {
            Text(label)
                .foregroundStyle(.primary)
            Spacer()
            Text(DilutionResult.formatVolume(microliters))
                .fontWeight(.semibold)
                .monospacedDigit()
        }
    }
}

// MARK: - Cross-platform background color helper

#if os(macOS)
extension Color {
    static let secondarySystemBackgroundCompat = Color(NSColor.controlBackgroundColor)
}
#else
extension Color {
    static let secondarySystemBackgroundCompat = Color(UIColor.secondarySystemBackground)
}
#endif

#Preview {
    ContentView()
}
