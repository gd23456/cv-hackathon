import Network
import Observation

/// Whether this iPhone has any network, so the no-network claim is visible and true.
@Observable
final class NetworkStatus {
    private(set) var isOffline = false
    private let monitor = NWPathMonitor()

    init() {
        monitor.pathUpdateHandler = { [weak self] path in
            let offline = path.status != .satisfied
            Task { @MainActor in self?.isOffline = offline }
        }
        monitor.start(queue: DispatchQueue(label: "network-status"))
    }
}
