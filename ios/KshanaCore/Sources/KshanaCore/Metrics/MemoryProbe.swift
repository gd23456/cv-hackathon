import Darwin
import Foundation

/// Memory figures as iOS accounts them: `phys_footprint` is what the system's
/// memory limit is enforced against. On iOS 26 it may not include all Neural
/// Engine memory, so always report it alongside the iOS version.
public enum MemoryProbe {
    public struct Snapshot: Sendable {
        public let footprintBytes: UInt64
        /// Highest footprint since the process started.
        public let lifetimePeakBytes: UInt64
    }

    public static func snapshot() -> Snapshot? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        return Snapshot(
            footprintBytes: UInt64(info.phys_footprint),
            lifetimePeakBytes: UInt64(info.ledger_phys_footprint_peak)
        )
    }

    /// Bytes this app may still allocate before the system's limit. iOS only.
    public static func availableBytes() -> UInt64? {
        #if os(iOS)
        UInt64(os_proc_available_memory())
        #else
        nil
        #endif
    }
}
