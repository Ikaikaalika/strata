import Foundation

/// The only service-to-host capability. The host validates every envelope
/// before relaying text or metering and invalidates the sink at task terminal.
@objc public protocol LokahiEventSinkV1: AnyObject {
    func receiveLokahiEnvelope(_ envelope: Data)
}

/// Dedicated bounded Lokahi RPCs. Do not route these messages through the
/// generic runtime profile that accepts entrypoints, arguments, or environments.
@objc public protocol LokahiComputeServiceV1: AnyObject {
    func startLokahi(
        _ request: Data,
        eventSink: LokahiEventSinkV1,
        reply: @escaping (Error?) -> Void
    )

    func controlLokahi(
        _ control: Data,
        reply: @escaping (Error?) -> Void
    )
}
