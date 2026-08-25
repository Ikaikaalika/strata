import Foundation

/// The only service-to-host capability. The host validates every envelope
/// before relaying text or metering and invalidates the sink at task terminal.
@objc public protocol StrataEventSinkV1: AnyObject {
    func receiveStrataEnvelope(_ envelope: Data)
}

/// Dedicated bounded Strata RPCs. Do not route these messages through the
/// generic runtime profile that accepts entrypoints, arguments, or environments.
@objc public protocol StrataComputeServiceV1: AnyObject {
    func startStrata(
        _ request: Data,
        eventSink: StrataEventSinkV1,
        reply: @escaping (Error?) -> Void
    )

    func controlStrata(
        _ control: Data,
        reply: @escaping (Error?) -> Void
    )
}
