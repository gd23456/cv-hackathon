import KshanaCore
import SwiftUI

/// Moving cyan, magenta and yellow ink on black, which quickens while generating.
struct AuroraBackground: View {
    var energy: Double = 0
    var animated = true

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30, paused: !animated)) { context in
            let t = Float(context.date.timeIntervalSinceReferenceDate.truncatingRemainder(dividingBy: 3600)) * 0.35
            let reach = Float(0.12 + 0.14 * energy)
            MeshGradient(
                width: 3, height: 3,
                points: [
                    [0, 0], [0.5 + reach * sin(t), 0], [1, 0],
                    [0, 0.5 + reach * cos(t * 0.8)], [0.5 + reach * cos(t * 1.2), 0.5 + reach * sin(t * 1.5)], [1, 0.5 + reach * sin(t * 0.9)],
                    [0, 1], [0.5 + reach * cos(t * 0.7), 1], [1, 1],
                ],
                colors: [
                    .black, Color(red: 0.0, green: 0.13, blue: 0.2), .black,
                    Color(red: 0.2, green: 0.0, blue: 0.13), Color(red: 0.05, green: 0.03, blue: 0.1), Color(red: 0.0, green: 0.16, blue: 0.24),
                    .black, Color(red: 0.24, green: 0.0, blue: 0.16), Color(red: 0.14, green: 0.12, blue: 0.0),
                ]
            )
        }
        .ignoresSafeArea()
        .accessibilityHidden(true)
    }
}

/// The name as three ink plates. They drift apart while working and meet as white when done.
struct InkTitle: View {
    let text: String
    let active: Bool

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 14, paused: !active)) { context in
            let t = context.date.timeIntervalSinceReferenceDate
            let spread: CGFloat = active ? 4 : 0
            ZStack {
                plate(Color(red: 0, green: 0.68, blue: 0.94), x: -spread * CGFloat(sin(t * 9)), y: spread * 0.4 * CGFloat(cos(t * 7)))
                plate(Color(red: 0.93, green: 0, blue: 0.55), x: spread * CGFloat(cos(t * 8)), y: -spread * 0.4 * CGFloat(sin(t * 11)))
                plate(Color(red: 1, green: 0.95, blue: 0), x: spread * 0.6 * CGFloat(sin(t * 13)), y: spread * 0.5 * CGFloat(cos(t * 5)))
                if !active { Text(text).foregroundStyle(.white) }
            }
            .font(.system(size: 34, weight: .black, design: .default).width(.expanded))
            .animation(.spring(duration: 0.4, bounce: 0.3), value: active)
        }
        .accessibilityLabel(text)
    }

    private func plate(_ color: Color, x: CGFloat, y: CGFloat) -> some View {
        Text(text).foregroundStyle(color).blendMode(.plusLighter).offset(x: x, y: y)
    }
}

/// One real candidate flying past the camera. Only images that passed the safety gate get here.
struct FlyingCard: Identifiable {
    let id: Int
    let image: UIImage
    let born: Date
    let angle: Double
    let spin: Double
}

struct WarpField: View {
    let cards: [FlyingCard]
    static let life = 1.05

    var body: some View {
        TimelineView(.animation) { context in
            ZStack {
                ForEach(cards) { card in
                    let t = context.date.timeIntervalSince(card.born) / Self.life
                    if t >= 0, t < 1 {
                        Image(uiImage: card.image)
                            .resizable()
                            .frame(width: 92, height: 92)
                            .clipShape(.rect(cornerRadius: 10))
                            .rotation3DEffect(.degrees(card.spin * t), axis: (x: sin(card.angle), y: cos(card.angle), z: 0.25), perspective: 0.5)
                            .scaleEffect(0.3 + t * t * 1.9)
                            .offset(x: cos(card.angle) * 330 * t * t, y: sin(card.angle) * 430 * t * t)
                            .opacity(min(t / 0.12, 1) * (1 - t * t))
                    }
                }
            }
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }
}

/// A beam sweeping the image while checks run.
struct ScanLine: View {
    var body: some View {
        GeometryReader { proxy in
            TimelineView(.animation) { context in
                let phase = context.date.timeIntervalSinceReferenceDate.truncatingRemainder(dividingBy: 1.3) / 1.3
                LinearGradient(colors: [.clear, Color(red: 0, green: 0.8, blue: 1).opacity(0.55), .white.opacity(0.9)], startPoint: .top, endPoint: .bottom)
                    .frame(height: 70)
                    .offset(y: proxy.size.height * phase - 70)
                    .blendMode(.plusLighter)
            }
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }
}

/// The detector's real boxes, drawing themselves in.
struct DetectionOverlay: View {
    let detections: [Detection]
    let revision: Int

    var body: some View {
        GeometryReader { proxy in
            ForEach(Array(detections.prefix(8).enumerated()), id: \.offset) { index, detection in
                DetectionBox(detection: detection, size: proxy.size, delay: Double(index) * 0.07)
            }
        }
        .id(revision)
        .allowsHitTesting(false)
    }
}

private struct DetectionBox: View {
    let detection: Detection
    let size: CGSize
    let delay: Double
    @State private var drawn = false

    var body: some View {
        let frame = CGRect(
            x: detection.box.minX * size.width, y: detection.box.minY * size.height,
            width: detection.box.width * size.width, height: detection.box.height * size.height
        )
        ZStack(alignment: .topLeading) {
            Rectangle()
                .trim(from: 0, to: drawn ? 1 : 0)
                .stroke(Color(red: 1, green: 0.95, blue: 0), lineWidth: 2)
            Text("\(detection.label) \(Int(detection.confidence * 100))%")
                .font(.caption2.weight(.bold).monospacedDigit())
                .foregroundStyle(.black)
                .padding(.horizontal, 5)
                .padding(.vertical, 2)
                .background(Color(red: 1, green: 0.95, blue: 0))
                .opacity(drawn ? 1 : 0)
                .fixedSize()
        }
        .frame(width: frame.width, height: frame.height)
        .position(x: frame.midX, y: frame.midY)
        .onAppear {
            withAnimation(.easeOut(duration: 0.45).delay(delay)) { drawn = true }
        }
    }
}

/// A burst of ink when the checks mostly pass. Positions come from a fixed formula.
struct InkBurst: View {
    let trigger: Int
    @State private var start: Date?
    private let life = 1.7
    private let colors: [Color] = [
        Color(red: 0, green: 0.68, blue: 0.94), Color(red: 0.93, green: 0, blue: 0.55),
        Color(red: 1, green: 0.95, blue: 0), .white,
    ]

    var body: some View {
        TimelineView(.animation(minimumInterval: nil, paused: start == nil)) { context in
            Canvas { canvas, size in
                guard let start else { return }
                let t = context.date.timeIntervalSince(start)
                guard t >= 0, t < life else { return }
                for i in 0..<110 {
                    let a = Double(i) * 2.399963
                    let speed = 180 + Double((i * 37) % 100) * 4.2
                    let x = size.width / 2 + cos(a) * speed * t
                    let y = size.height * 0.34 + sin(a) * speed * t + 620 * t * t
                    let side = 5 + Double(i % 5)
                    var piece = canvas
                    piece.translateBy(x: x, y: y)
                    piece.rotate(by: .radians(a + t * Double(4 + i % 7)))
                    piece.opacity = max(0, 1 - t / life)
                    piece.fill(Path(CGRect(x: -side / 2, y: -side / 4, width: side, height: side / 2)), with: .color(colors[i % colors.count]))
                }
            }
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
        .onChange(of: trigger) { start = .now }
    }
}

/// A shader reveal for the first image, and a ripple for each better one.
struct ImageArrival: ViewModifier {
    let revision: Int
    let enabled: Bool

    func body(content: Content) -> some View {
        content
            .keyframeAnimator(initialValue: 1.0, trigger: revision) { view, progress in
                view.visualEffect { [enabled, revision] effect, proxy in
                    effect.layerEffect(
                        ShaderLibrary.developReveal(.float2(proxy.size), .float(progress)),
                        maxSampleOffset: CGSize(width: 60, height: 60),
                        isEnabled: enabled && revision == 1 && progress < 1
                    )
                }
            } keyframes: { _ in
                MoveKeyframe(0.0)
                CubicKeyframe(1.0, duration: 0.55)
            }
            .keyframeAnimator(initialValue: 2.0, trigger: revision) { view, elapsed in
                view.visualEffect { [enabled, revision] effect, proxy in
                    effect.layerEffect(
                        ShaderLibrary.ripple(
                            .float2(proxy.size.width / 2, proxy.size.height / 2),
                            .float(elapsed), .float(16), .float(15), .float(7), .float(1100)
                        ),
                        maxSampleOffset: CGSize(width: 30, height: 30),
                        isEnabled: enabled && revision > 1 && elapsed < 1.2
                    )
                }
            } keyframes: { _ in
                MoveKeyframe(0.0)
                LinearKeyframe(1.3, duration: 1.3)
            }
    }
}

/// A rotating ink ring around the button while working, and a springy press.
struct InkButtonStyle: ButtonStyle {
    let working: Bool

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.headline)
            .foregroundStyle(.black)
            .frame(maxWidth: .infinity, minHeight: 54)
            .background(.white, in: .rect(cornerRadius: 16))
            .overlay {
                if working {
                    TimelineView(.animation) { context in
                        let angle = context.date.timeIntervalSinceReferenceDate * 320
                        RoundedRectangle(cornerRadius: 16)
                            .strokeBorder(
                                AngularGradient(
                                    colors: [Color(red: 0, green: 0.68, blue: 0.94), Color(red: 0.93, green: 0, blue: 0.55), Color(red: 1, green: 0.95, blue: 0), Color(red: 0, green: 0.68, blue: 0.94)],
                                    center: .center, angle: .degrees(angle)
                                ),
                                lineWidth: 4
                            )
                    }
                }
            }
            .scaleEffect(configuration.isPressed ? 0.96 : 1)
            .animation(.spring(duration: 0.25, bounce: 0.45), value: configuration.isPressed)
    }
}
