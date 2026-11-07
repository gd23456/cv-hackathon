import CoreML
import Testing
@testable import KshanaCore

@Test func `same seed gives the same noise`() {
    var a = SeededNormal(seed: 42)
    var b = SeededNormal(seed: 42)
    #expect(a.fill(count: 1000) == b.fill(count: 1000))
}

@Test func `different seeds give different noise`() {
    var a = SeededNormal(seed: 42)
    var b = SeededNormal(seed: 43)
    #expect(a.fill(count: 100) != b.fill(count: 100))
}

@Test func `noise is roughly standard normal`() {
    var source = SeededNormal(seed: 7)
    let values = source.fill(count: 100_000)
    let mean = values.reduce(0, +) / Float(values.count)
    let variance = values.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Float(values.count)
    #expect(abs(mean) < 0.02)
    #expect(abs(variance - 1) < 0.03)
}

@Test func `long prompts are cut to 77 tokens and keep the end token`() {
    let fitted = Generator.fit(tokenIDs: Array(1...100), endToken: 49407)
    #expect(fitted.ids.count == 77)
    #expect(fitted.ids.last == 49407)
    #expect(fitted.tokenCount == 100)
    #expect(fitted.wasTruncated)
}

@Test func `short prompts are left alone`() {
    let fitted = Generator.fit(tokenIDs: Array(1...77), endToken: 49407)
    #expect(fitted.ids.last == 77)
    #expect(!fitted.wasTruncated)
}

@Test func `output reader respects row padding`() throws {
    // 1 channel of 2x2 values, with a padded stride of 3 between rows.
    let storage = UnsafeMutablePointer<Float32>.allocate(capacity: 6)
    defer { storage.deallocate() }
    for (i, value) in [Float32(1), 2, -9, 3, 4, -9].enumerated() { storage[i] = value }
    let array = try MLMultiArray(
        dataPointer: storage, shape: [1, 1, 2, 2], dataType: .float32,
        strides: [6, 6, 3, 1]
    )
    #expect(try Generator.floats(from: array) == [1, 2, 3, 4])
}

@Test func `decoder output maps minus one to one onto 0 to 255`() throws {
    let array = try MLMultiArray(shape: [1, 3, 1, 2], dataType: .float32)
    array.withUnsafeMutableBufferPointer(ofType: Float32.self) { buffer, _ in
        for (i, value) in [Float32(-1), 1, -2, 2, 0, 0].enumerated() { buffer[i] = value }
    }
    let image = try Generator.rgba(from: array)
    #expect(image.pixels == [0, 0, 128, 255, 255, 255, 128, 255])
}

@Test func `non finite decoder output is an error`() throws {
    let array = try MLMultiArray(shape: [1, 3, 1, 1], dataType: .float32)
    array.withUnsafeMutableBufferPointer(ofType: Float32.self) { buffer, _ in
        buffer[0] = .nan
    }
    #expect(throws: GeneratorError.self) { try Generator.rgba(from: array) }
}

@Test func `blending noise keeps it at full strength and hits both ends`() {
    var a = SeededNormal(seed: 1), b = SeededNormal(seed: 2)
    let x = a.fill(count: 16384), y = b.fill(count: 16384)
    func length(_ v: [Float]) -> Float { v.reduce(0) { $0 + $1 * $1 }.squareRoot() }
    let middle = SeededNormal.slerp(x, y, t: 0.5)
    #expect(abs(length(middle) - length(x)) / length(x) < 0.02)
    #expect(zip(SeededNormal.slerp(x, y, t: 0), x).allSatisfy { abs($0 - $1) < 1e-3 })
    #expect(zip(SeededNormal.slerp(x, y, t: 1), y).allSatisfy { abs($0 - $1) < 1e-3 })
}
