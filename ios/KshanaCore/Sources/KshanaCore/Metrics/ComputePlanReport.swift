import CoreML
import Foundation

/// Where Core ML plans to run a model's operations on this device.
public struct ComputePlanReport: Sendable {
    public let model: String
    public let neuralEngineOps: Int
    public let gpuOps: Int
    public let cpuOps: Int
    public let unknownOps: Int
    /// Non-trivial operations pinned to the CPU, most frequent first.
    public let cpuOpNames: [String: Int]

    public var totalOps: Int { neuralEngineOps + gpuOps + cpuOps + unknownOps }

    public static func load(modelAt url: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine) async throws -> ComputePlanReport {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        let plan = try await MLComputePlan.load(contentsOf: url, configuration: config)
        guard case .program(let program) = plan.modelStructure, let main = program.functions["main"] else {
            throw GeneratorError.badOutput("no main function in \(url.lastPathComponent)")
        }

        var ne = 0, gpu = 0, cpu = 0, unknown = 0
        var cpuNames: [String: Int] = [:]

        func visit(_ block: MLModelStructure.Program.Block) {
            for operation in block.operations {
                for nested in operation.blocks { visit(nested) }
                // Constants carry weights and never execute on a compute device.
                if operation.operatorName == "const" { continue }
                switch plan.deviceUsage(for: operation)?.preferred {
                case .neuralEngine: ne += 1
                case .gpu: gpu += 1
                case .cpu:
                    cpu += 1
                    cpuNames[operation.operatorName, default: 0] += 1
                default: unknown += 1
                }
            }
        }
        visit(main.block)

        return ComputePlanReport(
            model: url.deletingPathExtension().lastPathComponent,
            neuralEngineOps: ne, gpuOps: gpu, cpuOps: cpu, unknownOps: unknown,
            cpuOpNames: cpuNames
        )
    }
}
