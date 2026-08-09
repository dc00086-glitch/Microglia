import SwiftUI

struct LearnView: View {
    var body: some View {
        NavigationStack {
            List(LessonLibrary.all) { lesson in
                NavigationLink(value: lesson.id) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(lesson.title).font(.headline)
                        Text(lesson.summary)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 4)
                }
            }
            .navigationTitle("Learn Blackjack")
            .navigationDestination(for: UUID.self) { id in
                if let lesson = LessonLibrary.all.first(where: { $0.id == id }) {
                    LessonDetailView(lesson: lesson)
                }
            }
        }
    }
}

struct LessonDetailView: View {
    let lesson: Lesson

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                ForEach(Array(paragraphs.enumerated()), id: \.offset) { _, block in
                    BlockText(block)
                }
            }
            .padding()
        }
        .navigationTitle(lesson.title)
        .navigationBarTitleDisplayMode(.inline)
    }

    private var paragraphs: [String] {
        lesson.body.components(separatedBy: "\n\n")
    }
}

/// Renders a block of text, turning lines that start with "•" into a bullet list
/// and applying inline **bold** markdown.
private struct BlockText: View {
    let text: String
    init(_ text: String) { self.text = text }

    var body: some View {
        let lines = text.components(separatedBy: "\n")
        VStack(alignment: .leading, spacing: 6) {
            ForEach(Array(lines.enumerated()), id: \.offset) { _, line in
                Text(attributed(line))
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func attributed(_ line: String) -> AttributedString {
        (try? AttributedString(markdown: line,
            options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)))
            ?? AttributedString(line)
    }
}
