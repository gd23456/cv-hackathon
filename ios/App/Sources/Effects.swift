import KshanaCore
import SwiftUI

extension Color {
    static let kshanaBackground = Color(red: 0.05, green: 0.06, blue: 0.075)
    static let kshanaPanel = Color(red: 0.09, green: 0.105, blue: 0.125)
    static let kshanaAccent = Color(red: 0.31, green: 0.58, blue: 1.0)
    /// Inset controls on glass, the same as the website's.
    static let kshanaWell = Color.black.opacity(0.36)
    static let kshanaKey = LinearGradient(
        colors: [Color(red: 0.56, green: 0.73, blue: 1), .kshanaAccent, Color(red: 0.18, green: 0.44, blue: 0.85)],
        startPoint: .top, endPoint: .bottom
    )
}

/// The website's marbled desk, with its dark wash and dot grid baked in. Deliberately still.
struct DeskBackground: View {
    var body: some View {
        Color.kshanaBackground
            .overlay { Image(decorative: "Desk").resizable().scaledToFill() }
            .clipped()
            .ignoresSafeArea()
            .accessibilityHidden(true)
    }
}

/// A folder outline with a tab on one side, the same curve as the website's folder.js.
nonisolated struct FolderShape: Shape {
    var tabWidth: CGFloat
    var tabHeight: CGFloat
    var radius: CGFloat = 20
    var tabOnRight = false

    func path(in rect: CGRect) -> Path {
        let w = rect.width, h = rect.height, r = radius, tab = tabHeight
        let shoulder = tab * 1.35
        var path = Path()
        if tabOnRight {
            let x = w - tabWidth
            path.move(to: CGPoint(x: 0, y: tab + r))
            path.addArc(tangent1End: CGPoint(x: 0, y: tab), tangent2End: CGPoint(x: r, y: tab), radius: r)
            path.addLine(to: CGPoint(x: x, y: tab))
            path.addCurve(to: CGPoint(x: x + shoulder, y: 0), control1: CGPoint(x: x + shoulder / 2, y: tab), control2: CGPoint(x: x + shoulder / 2, y: 0))
            path.addArc(tangent1End: CGPoint(x: w, y: 0), tangent2End: CGPoint(x: w, y: r), radius: r)
        } else {
            let end = tabWidth
            path.move(to: CGPoint(x: 0, y: r))
            path.addArc(tangent1End: .zero, tangent2End: CGPoint(x: r, y: 0), radius: r)
            path.addLine(to: CGPoint(x: end - shoulder, y: 0))
            path.addCurve(to: CGPoint(x: end, y: tab), control1: CGPoint(x: end - shoulder / 2, y: 0), control2: CGPoint(x: end - shoulder / 2, y: tab))
            path.addArc(tangent1End: CGPoint(x: w, y: tab), tangent2End: CGPoint(x: w, y: tab + r), radius: r)
        }
        path.addArc(tangent1End: CGPoint(x: w, y: h), tangent2End: CGPoint(x: w - r, y: h), radius: r)
        path.addArc(tangent1End: CGPoint(x: 0, y: h), tangent2End: CGPoint(x: 0, y: h - r), radius: r)
        path.closeSubpath()
        return path.offsetBy(dx: rect.minX, dy: rect.minY)
    }
}

extension View {
    /// Smoked glass cut to a shape, with a bright rim along the top.
    func glass(in shape: some Shape) -> some View {
        background {
            shape.fill(.ultraThinMaterial).overlay { shape.fill(.black.opacity(0.26)) }
        }
        .overlay {
            shape.stroke(
                LinearGradient(colors: [.white.opacity(0.5), .white.opacity(0.16), .white.opacity(0.08)], startPoint: .top, endPoint: .bottom),
                lineWidth: 1
            )
            .allowsHitTesting(false)
        }
    }

    /// A folder's label, sitting in its tab.
    func folderTab(_ title: String, height: CGFloat) -> some View {
        overlay(alignment: .topLeading) {
            Text(title)
                .font(.caption.weight(.semibold))
                .frame(height: height)
                .padding(.leading, 18)
                .accessibilityHidden(true)
        }
    }

    /// Frosted rows for the lists on the Inside and Benchmarks tabs.
    func glassRows() -> some View {
        listRowBackground(Rectangle().fill(.ultraThinMaterial))
    }
}

/// A better image drops into the folder, as on the website.
struct DropIn: ViewModifier {
    let revision: Int
    let enabled: Bool

    nonisolated struct Pose: Sendable {
        var y: CGFloat = 0
        var angle: Double = 0
    }

    func body(content: Content) -> some View {
        content.keyframeAnimator(initialValue: Pose(), trigger: revision) { [enabled] view, pose in
            view
                .offset(y: enabled ? pose.y : 0)
                .rotationEffect(.degrees(enabled ? pose.angle : 0))
        } keyframes: { _ in
            KeyframeTrack(\.y) {
                MoveKeyframe(-22)
                CubicKeyframe(6, duration: 0.24)
                CubicKeyframe(0, duration: 0.16)
            }
            KeyframeTrack(\.angle) {
                MoveKeyframe(-1.6)
                CubicKeyframe(0, duration: 0.4)
            }
        }
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
                            .clipShape(.rect(cornerRadius: 8))
                            .rotation3DEffect(.degrees(card.spin * t), axis: (x: sin(card.angle), y: cos(card.angle), z: 0.25), perspective: 0.5)
                            .scaleEffect(0.3 + t * t * 1.9)
                            .offset(x: cos(card.angle) * 330 * t * t, y: sin(card.angle) * 430 * t * t)
                            .opacity(min(t / 0.12, 1) * (1 - t * t) * 0.8)
                    }
                }
            }
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }
}

/// A thin beam sweeping the image while checks run.
struct ScanLine: View {
    var body: some View {
        GeometryReader { proxy in
            TimelineView(.animation) { context in
                let phase = context.date.timeIntervalSinceReferenceDate.truncatingRemainder(dividingBy: 1.4) / 1.4
                LinearGradient(colors: [.clear, Color.kshanaAccent.opacity(0.35)], startPoint: .top, endPoint: .bottom)
                    .frame(height: 44)
                    .overlay(alignment: .bottom) { Rectangle().fill(.white.opacity(0.7)).frame(height: 1) }
                    .offset(y: proxy.size.height * phase - 44)
                    .blendMode(.plusLighter)
            }
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }
}

/// The same animation as the website while the first image is on its way:
/// a dot field lit by two drifting pools of light.
struct GeneratingView: View {
    let prompt: String
    let animated: Bool
    @State private var shine = false

    var body: some View {
        VStack(spacing: 12) {
            ZStack(alignment: .bottomTrailing) {
                TimelineView(.animation(minimumInterval: 1.0 / 24, paused: !animated)) { context in
                    Canvas { canvas, size in
                        let t = context.date.timeIntervalSinceReferenceDate
                        let centres = [
                            CGPoint(x: size.width * (0.5 + 0.34 * cos(t * 0.7)), y: size.height * (0.5 + 0.3 * sin(t * 0.9))),
                            CGPoint(x: size.width * (0.5 + 0.3 * sin(t * 0.5 + 2)), y: size.height * (0.5 + 0.34 * cos(t * 0.6 + 1))),
                        ]
                        let reach = size.width * 0.34
                        let spacing: CGFloat = 11
                        var y = spacing / 2
                        while y < size.height {
                            var x = spacing / 2
                            while x < size.width {
                                let light = centres.reduce(0.0) { total, c in
                                    let d = hypot(x - c.x, y - c.y) / reach
                                    return total + exp(-Double(d * d) * 1.6)
                                }
                                let dot = CGRect(x: x - 1.2, y: y - 1.2, width: 2.4, height: 2.4)
                                canvas.fill(Path(ellipseIn: dot), with: .color(Color(red: 0.81, green: 0.88, blue: 1).opacity(min(1, 0.08 + light))))
                                x += spacing
                            }
                            y += spacing
                        }
                    }
                }
                .background(Color.kshanaBackground)
                Text("512 × 512")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 7)
                    .padding(.vertical, 3)
                    .background(.black.opacity(0.55), in: .rect(cornerRadius: 6))
                    .padding(8)
            }
            .clipShape(.rect(cornerRadius: 14))
            .frame(maxWidth: 240, maxHeight: 240)
            Text("Generating image")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.secondary)
                .overlay {
                    GeometryReader { proxy in
                        LinearGradient(colors: [.clear, .white, .clear], startPoint: .leading, endPoint: .trailing)
                            .frame(width: proxy.size.width * 0.6)
                            .offset(x: shine ? proxy.size.width : -proxy.size.width * 0.6)
                    }
                    .mask { Text("Generating image").font(.subheadline.weight(.semibold)) }
                }
            Text("\u{201C}\(prompt)\u{201D}")
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .lineLimit(2)
        }
        .padding(18)
        .onAppear {
            guard animated else { return }
            withAnimation(.linear(duration: 1.6).repeatForever(autoreverses: false)) { shine = true }
        }
        .accessibilityElement(children: .combine)
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
                            .float(elapsed), .float(9), .float(15), .float(8), .float(1100)
                        ),
                        maxSampleOffset: CGSize(width: 20, height: 20),
                        isEnabled: enabled && revision > 1 && elapsed < 1.1
                    )
                }
            } keyframes: { _ in
                MoveKeyframe(0.0)
                LinearKeyframe(1.2, duration: 1.2)
            }
    }
}
