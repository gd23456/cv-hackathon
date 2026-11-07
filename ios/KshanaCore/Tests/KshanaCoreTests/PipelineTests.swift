import CoreGraphics
import Foundation
import Testing
@testable import KshanaCore

private func solid(_ r: UInt8, _ g: UInt8, _ b: UInt8, size: Int = 64) -> RGBAImage {
    var pixels = [UInt8](repeating: 255, count: size * size * 4)
    for i in 0..<(size * size) {
        pixels[i * 4] = r
        pixels[i * 4 + 1] = g
        pixels[i * 4 + 2] = b
    }
    return RGBAImage(width: size, height: size, pixels: pixels)
}

private func gradient(size: Int = 64) -> RGBAImage {
    var pixels = [UInt8](repeating: 255, count: size * size * 4)
    for y in 0..<size {
        for x in 0..<size {
            let value = UInt8((x + y) * 255 / (2 * size))
            let i = (y * size + x) * 4
            pixels[i] = value
            pixels[i + 1] = value
            pixels[i + 2] = value
        }
    }
    return RGBAImage(width: size, height: size, pixels: pixels)
}

private let calm = DeviceState(thermal: .nominal, lowPower: false, availableMB: 2000)

@Suite struct PromptParserTests {
    @Test func `objects colours and relation`() {
        let parsed = PromptParser.parse("a red apple to the left of a blue cup on a wooden table")
        #expect(parsed.objects.first?.label == "apple")
        #expect(parsed.objects.first?.color == "red")
        #expect(parsed.objects.dropFirst().first?.label == "cup")
        #expect(parsed.objects.dropFirst().first?.color == "blue")
        #expect(parsed.relations.first == ParsedRelation(subject: 0, relation: .leftOf, object: 1))
        #expect(parsed.namesColor)
    }

    @Test func `number becomes a count`() {
        let parsed = PromptParser.parse("two dogs sitting on a couch")
        #expect(parsed.objects.first?.label == "dog")
        #expect(parsed.objects.first?.count == 2)
    }

    @Test func `synonyms map to detector classes`() {
        #expect(PromptParser.parse("a puppy").objects.first?.label == "dog")
        #expect(PromptParser.parse("a coffee mug").objects.contains { $0.label == "cup" })
    }

    @Test func `objects the detector cannot see stay unverifiable`() {
        let parsed = PromptParser.parse("a mountain at sunset")
        #expect(parsed.objects.contains { $0.noun == "mountain" && $0.label == nil })
    }

    @Test func `rewordings name the target first and keep the scene`() {
        let text = "a red apple to the left of a blue cup on a wooden table"
        let lines = PromptParser.rewordings(of: text, parsed: PromptParser.parse(text))
        #expect(lines.first == "a blue cup, a red apple beside it on the left, on a wooden table")
        #expect(lines.count == 3)
    }

    @Test func `simple prompts aren't reworded`() {
        #expect(PromptParser.rewordings(of: "a dog on the grass", parsed: PromptParser.parse("a dog on the grass")).isEmpty)
        let two = "a cat and a dog"
        #expect(PromptParser.rewordings(of: two, parsed: PromptParser.parse(two)).isEmpty)
    }

    @Test func `harder prompts score as more complex`() {
        let simple = PromptParser.parse("a dog").complexity
        let hard = PromptParser.parse("a red apple to the left of a blue cup").complexity
        #expect(hard > simple)
    }

    @Test(arguments: ["", "   ", "!!!", "12345", "こんにちは", String(repeating: "dog ", count: 500)])
    func `odd input never crashes`(input: String) {
        _ = PromptParser.parse(input)
    }
}

@Suite struct ChecklistTests {
    private let parsed = PromptParser.parse("a red apple to the left of a blue cup")

    private func detections(appleX: CGFloat, cupX: CGFloat) -> [Detection] {
        [
            Detection(label: "apple", confidence: 0.8, box: CGRect(x: appleX, y: 0.4, width: 0.2, height: 0.2)),
            Detection(label: "cup", confidence: 0.8, box: CGRect(x: cupX, y: 0.4, width: 0.2, height: 0.2)),
        ]
    }

    @Test func `correct position passes and swapped position fails`() {
        let good = Checklist.evaluate(parsed, detections: detections(appleX: 0.1, cupX: 0.6), image: solid(0, 0, 0))
        #expect(good.contains { $0.title.contains("left of") && $0.state == .passed })
        let bad = Checklist.evaluate(parsed, detections: detections(appleX: 0.6, cupX: 0.1), image: solid(0, 0, 0))
        #expect(bad.contains { $0.title.contains("left of") && $0.state == .failed })
    }

    @Test func `missing object fails`() {
        let results = Checklist.evaluate(parsed, detections: [], image: solid(0, 0, 0))
        #expect(results.contains { $0.title == "apple" && $0.state == .failed })
    }

    @Test func `colour check reads the pixels in the box`() {
        let box = CGRect(x: 0.2, y: 0.2, width: 0.6, height: 0.6)
        #expect(Checklist.colorShare(of: "red", in: box, image: solid(220, 20, 20)) > 0.9)
        #expect(Checklist.colorShare(of: "blue", in: box, image: solid(220, 20, 20)) == 0)
        #expect(Checklist.colorShare(of: "blue", in: box, image: solid(20, 40, 220)) > 0.9)
    }

    @Test func `a half red cup fails the blue check when the prompt names both`() {
        var pixels = [UInt8](repeating: 255, count: 64 * 64 * 4)
        for y in 0..<64 {
            for x in 0..<64 {
                let i = (y * 64 + x) * 4
                let blue = x >= 32
                pixels[i] = blue ? 30 : 220
                pixels[i + 1] = 40
                pixels[i + 2] = blue ? 220 : 30
            }
        }
        let image = RGBAImage(width: 64, height: 64, pixels: pixels)
        let box = CGRect(x: 0, y: 0, width: 1, height: 1)
        let prompt = PromptParser.parse("a red apple and a blue cup")
        let results = Checklist.evaluate(prompt, detections: [Detection(label: "cup", confidence: 0.9, box: box)], image: image)
        #expect(results.contains { $0.title == "blue cup" && $0.state == .failed })
    }

    @Test func `heavily overlapping objects aren't beside each other`() {
        let a = CGRect(x: 0.1, y: 0.1, width: 0.5, height: 0.5)
        let b = CGRect(x: 0.25, y: 0.1, width: 0.5, height: 0.5)
        #expect(!Checklist.satisfies(.leftOf, a, b))
        #expect(Checklist.satisfies(.leftOf, a, CGRect(x: 0.65, y: 0.1, width: 0.3, height: 0.5)))
    }

    @Test func `unknown objects are not verifiable rather than passed`() {
        let results = Checklist.evaluate(PromptParser.parse("a mountain"), detections: [], image: solid(0, 0, 0))
        #expect(results.allSatisfy { $0.state == .notVerifiable })
    }
}

@Suite struct ClassicalTests {
    @Test func `flat image is blank and a gradient is not`() {
        #expect(ImageQuality.measure(solid(120, 120, 120)).isBlank)
        #expect(!ImageQuality.measure(gradient()).isBlank)
    }

    @Test func `gray world reduces a colour cast`() {
        let tinted = solid(200, 120, 120)
        let before = ImageQuality.measure(tinted).colorCast
        let after = ImageQuality.measure(ImageOps.grayWorld(tinted)).colorCast
        #expect(after < before)
    }

    @Test func `CLAHE raises the contrast of a dull image`() {
        var dull = gradient()
        for i in 0..<(dull.width * dull.height) {
            for c in 0..<3 { dull.pixels[i * 4 + c] = dull.pixels[i * 4 + c] / 8 + 100 }
        }
        let before = ImageQuality.measure(dull).contrast
        let after = ImageQuality.measure(ImageOps.clahe(dull)).contrast
        #expect(after > before)
    }

    @Test func `grid detector fires on an 8 pixel grid only`() {
        var grid = gradient()
        for y in 0..<grid.height {
            for x in stride(from: 0, to: grid.width, by: 8) {
                let i = (y * grid.width + x) * 4
                for c in 0..<3 { grid.pixels[i + c] = 255 }
            }
        }
        #expect(ImageQuality.measure(grid).gridScore > ImageQuality.measure(gradient()).gridScore)
    }

    @Test func `colour step is skipped when the prompt names colours`() {
        let stages = ClassicalPipeline.process(
            solid(200, 120, 120), quality: ImageQuality.measure(solid(200, 120, 120)),
            prompt: PromptParser.parse("a red apple")
        )
        #expect(stages.first?.applied == false)
        #expect(stages.first?.image == solid(200, 120, 120))
    }

    @Test func `edge aware sharpen keeps flat areas flat`() {
        #expect(ImageOps.edgeAwareSharpen(solid(90, 90, 90)) == solid(90, 90, 90))
    }

    @Test func `resize produces the requested size`() throws {
        let big = try ImageOps.resized(gradient(), width: 128, height: 128)
        #expect(big.width == 128)
        #expect(big.pixels.count == 128 * 128 * 4)
    }
}

@Suite struct PlannerTests {
    @Test func `fast is one image with no checks`() {
        let plan = Planner.decide(budget: .fast, complexity: 5, device: calm)
        #expect(plan.maxCandidates == 1)
        #expect(!plan.useVerifier)
    }

    @Test func `auto searches longer for complex prompts`() {
        let simple = Planner.decide(budget: .auto, complexity: 1, device: calm)
        let hard = Planner.decide(budget: .auto, complexity: 6, device: calm)
        #expect(hard.deadlineMs > simple.deadlineMs)
    }

    @Test func `low power halves the search and says so`() {
        let plan = Planner.decide(budget: .best, complexity: 1, device: DeviceState(thermal: .nominal, lowPower: true, availableMB: 2000))
        #expect(plan.deadlineMs == 2500)
        #expect(plan.reason.contains("Low Power"))
    }

    @Test func `heat reduces candidates and critical heat turns checks off`() {
        let hot = Planner.decide(budget: .best, complexity: 1, device: DeviceState(thermal: .serious, lowPower: false, availableMB: 2000))
        #expect(hot.maxCandidates == 2)
        let critical = Planner.decide(budget: .best, complexity: 1, device: DeviceState(thermal: .critical, lowPower: false, availableMB: 2000))
        #expect(critical.maxCandidates == 1)
        #expect(!critical.useVerifier)
    }

    @Test func `low memory turns checks off`() {
        let plan = Planner.decide(budget: .best, complexity: 1, device: DeviceState(thermal: .nominal, lowPower: false, availableMB: 200))
        #expect(!plan.useVerifier)
        #expect(plan.reason.contains("memory"))
    }

    @Test func `passing checks outrank alignment`() {
        let quality = ImageQuality.measure(gradient())
        let pass = [CheckResult(title: "cup", state: .passed)]
        let fail = [CheckResult(title: "cup", state: .failed)]
        #expect(Planner.score(checks: pass, alignment: 0.1, quality: quality) > Planner.score(checks: fail, alignment: 0.4, quality: quality))
    }

    @Test(arguments: ["an ass", "a big penis", "p e n i s", "p.e.n.i.s", "p-e-n-i-s", "pen1s", "b00bs", "bigpenisphoto", "n@ked woman", "nangi ladki", "woman with no clothes", "A NUDE person", "a girl in a swimsuit"])
    func `unsafe prompts are blocked`(prompt: String) {
        #expect(PromptSafety.isBlocked(prompt))
    }

    @Test(arguments: ["a class of students", "a glass of water", "an assassin in a cloak", "a bass guitar", "a cocktail on a table", "data analysis chart", "a sussex spaniel", "chicken breast on a plate", "a girl reading a book", "a butterfly on a flower", "open issue tracker on a laptop", "pass the ball", "titanic ship"])
    func `ordinary prompts are allowed`(prompt: String) {
        #expect(!PromptSafety.isBlocked(prompt))
    }

    @Test func `text the filter can't read is refused`() {
        #expect(PromptSafety.isUnreadable("नंगी औरत"))
        #expect(PromptSafety.isUnreadable("🍑🍆"))
        #expect(!PromptSafety.isUnreadable("a café in Paris"))
    }

    @Test func `the gate blocks everything until its model is loaded`() async {
        let gate = SafetyGate(modelURL: nil)
        #expect(await gate.allows(gradient()) == false)
    }

    @Test func `skin ratio separates skin tones from other colours`() {
        #expect(SafetyGate.skinRatio(solid(224, 172, 140)) > 0.9)
        #expect(SafetyGate.skinRatio(solid(20, 40, 220)) < 0.1)
    }
}
