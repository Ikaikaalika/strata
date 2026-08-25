import Foundation

public enum StrataWireContractError: Error, Equatable {
    case invalidJSON(String)
    case duplicateKey(String)
    case messageTooLarge(String)
    case invalidKeys(String)
    case invalidValue(String)
    case streamViolation(String)
}

public enum StrataWorkloadClassV1: String, Codable, Sendable {
    case interactive, throughput, background
}

public enum StrataMessageRoleV1: String, Codable, Sendable {
    case system, user, assistant
}

public enum StrataControlActionV1: String, Codable, Sendable {
    case cancel, drain
}

public enum StrataEventKindV1: String, Codable, Sendable {
    case accepted, progress
    case textDelta = "text_delta"
    case queueState = "queue_state"
    case warning
}

public enum StrataReceiptStatusV1: String, Codable, Sendable {
    case completed, failed, cancelled, rejected
}

public enum StrataFinishReasonV1: String, Codable, Sendable {
    case stop, length, cancelled, error, rejected
}

public enum StrataFailureStageV1: String, Codable, Sendable {
    case notStarted = "not_started"
    case running
    case partialOutput = "partial_output"
}

public enum StrataBackendV1: String, Codable, Sendable {
    case mlx, metal, coreml, ane, cpu, hybrid
}

public enum StrataRouteV1: String, Codable, Sendable {
    case batchGenerator = "batch_generator"
    case directSingleSequence = "direct_single_sequence"
    case nativePhaseProgram = "native_phase_program"
}

public struct StrataModelV1: Codable, Equatable, Sendable {
    public let id: String
    public let revision: String
    public let artifactDigest: String
    public let adapterID: String
    public let quantization: String

    enum CodingKeys: String, CodingKey {
        case id, revision, quantization
        case artifactDigest = "artifact_digest"
        case adapterID = "adapter_id"
    }
}

public struct StrataChatMessageV1: Codable, Equatable, Sendable {
    public let role: StrataMessageRoleV1
    public let content: String
}

public struct StrataSamplingV1: Codable, Equatable, Sendable {
    public let temperature: Double
    public let topP: Double
    public let seed: Int64?
    public let stop: [String]

    enum CodingKeys: String, CodingKey {
        case temperature, seed, stop
        case topP = "top_p"
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(temperature, forKey: .temperature)
        try container.encode(topP, forKey: .topP)
        if let seed {
            try container.encode(seed, forKey: .seed)
        } else {
            try container.encodeNil(forKey: .seed)
        }
        try container.encode(stop, forKey: .stop)
    }
}

public struct StrataObjectiveV1: Codable, Equatable, Sendable {
    public let workloadClass: StrataWorkloadClassV1
    public let contextTokens: Int
    public let maxOutputTokens: Int
    public let batchSize: Int
    public let targetTTFTMilliseconds: Double?
    public let minimumDecodeTokensPerSecond: Double?

    enum CodingKeys: String, CodingKey {
        case workloadClass = "workload_class"
        case contextTokens = "context_tokens"
        case maxOutputTokens = "max_output_tokens"
        case batchSize = "batch_size"
        case targetTTFTMilliseconds = "target_ttft_ms"
        case minimumDecodeTokensPerSecond = "min_decode_tokens_per_second"
    }
}

public struct StrataMemoryPolicyV1: Codable, Equatable, Sendable {
    public let maxRuntimeMemoryBytes: Int64
    public let maxResidentWeightBytes: Int64?
    public let ssdOffload: String

    enum CodingKeys: String, CodingKey {
        case maxRuntimeMemoryBytes = "max_runtime_memory_bytes"
        case maxResidentWeightBytes = "max_resident_weight_bytes"
        case ssdOffload = "ssd_offload"
    }
}

public struct StrataStartRequestV1: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let messageType: String
    public let requestID: String
    public let taskID: String
    public let operation: String
    public let runtimeRevision: String
    public let createdAtUnixMilliseconds: Int64
    public let deadlineUnixMilliseconds: Int64
    public let model: StrataModelV1
    public let messages: [StrataChatMessageV1]
    public let sampling: StrataSamplingV1
    public let objective: StrataObjectiveV1
    public let memory: StrataMemoryPolicyV1
    public let deploymentMode: String

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case messageType = "message_type"
        case requestID = "request_id"
        case taskID = "task_id"
        case operation
        case runtimeRevision = "runtime_revision"
        case createdAtUnixMilliseconds = "created_at_unix_ms"
        case deadlineUnixMilliseconds = "deadline_unix_ms"
        case model, messages, sampling, objective, memory
        case deploymentMode = "deployment_mode"
    }
}

public struct StrataControlV1: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let messageType: String
    public let controlID: String
    public let action: StrataControlActionV1
    public let targetRequestID: String?
    public let issuedAtUnixMilliseconds: Int64

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case messageType = "message_type"
        case controlID = "control_id"
        case action
        case targetRequestID = "target_request_id"
        case issuedAtUnixMilliseconds = "issued_at_unix_ms"
    }
}

public struct StrataEventV1: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let messageType: String
    public let requestID: String
    public let sequence: Int64
    public let kind: StrataEventKindV1
    public let emittedAtUnixMilliseconds: Int64
    public let progress: Double?
    public let textDelta: String?
    public let generatedTokensDelta: Int?
    public let queuePosition: Int?
    public let queueDepth: Int?
    public let message: String?

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case messageType = "message_type"
        case requestID = "request_id"
        case sequence, kind
        case emittedAtUnixMilliseconds = "emitted_at_unix_ms"
        case progress
        case textDelta = "text_delta"
        case generatedTokensDelta = "generated_tokens_delta"
        case queuePosition = "queue_position"
        case queueDepth = "queue_depth"
        case message
    }
}

public struct StrataUsageV1: Codable, Equatable, Sendable {
    public let promptTokens: Int
    public let completionTokens: Int
    public let totalTokens: Int

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
    }
}

public struct StrataMetricsV1: Codable, Equatable, Sendable {
    public let queueMilliseconds: Double
    public let ttftMilliseconds: Double?
    public let interTokenLatencyMilliseconds: Double?
    public let decodeTokensPerSecond: Double?
    public let peakMemoryBytes: Int64
    public let maximumBatchSize: Int

    enum CodingKeys: String, CodingKey {
        case queueMilliseconds = "queue_ms"
        case ttftMilliseconds = "ttft_ms"
        case interTokenLatencyMilliseconds = "inter_token_latency_ms"
        case decodeTokensPerSecond = "decode_tokens_per_second"
        case peakMemoryBytes = "peak_memory_bytes"
        case maximumBatchSize = "maximum_batch_size"
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(queueMilliseconds, forKey: .queueMilliseconds)
        if let ttftMilliseconds {
            try container.encode(ttftMilliseconds, forKey: .ttftMilliseconds)
        } else {
            try container.encodeNil(forKey: .ttftMilliseconds)
        }
        if let interTokenLatencyMilliseconds {
            try container.encode(interTokenLatencyMilliseconds, forKey: .interTokenLatencyMilliseconds)
        } else {
            try container.encodeNil(forKey: .interTokenLatencyMilliseconds)
        }
        if let decodeTokensPerSecond {
            try container.encode(decodeTokensPerSecond, forKey: .decodeTokensPerSecond)
        } else {
            try container.encodeNil(forKey: .decodeTokensPerSecond)
        }
        try container.encode(peakMemoryBytes, forKey: .peakMemoryBytes)
        try container.encode(maximumBatchSize, forKey: .maximumBatchSize)
    }
}

public struct StrataExecutionV1: Codable, Equatable, Sendable {
    public let backend: StrataBackendV1
    public let route: StrataRouteV1
    public let planID: String
    public let evidenceIDs: [String]
    public let fallbackUsed: Bool
    public let fallbackReason: String?

    enum CodingKeys: String, CodingKey {
        case backend, route
        case planID = "plan_id"
        case evidenceIDs = "evidence_ids"
        case fallbackUsed = "fallback_used"
        case fallbackReason = "fallback_reason"
    }
}

public struct StrataFailureV1: Codable, Equatable, Sendable {
    public let code: String
    public let stage: StrataFailureStageV1
    public let retryable: Bool
    public let message: String
}

public struct StrataReceiptV1: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let messageType: String
    public let requestID: String
    public let terminalSequence: Int64
    public let status: StrataReceiptStatusV1
    public let finishReason: StrataFinishReasonV1
    public let completedAtUnixMilliseconds: Int64
    public let outputText: String?
    public let usage: StrataUsageV1
    public let metrics: StrataMetricsV1
    public let execution: StrataExecutionV1
    public let failure: StrataFailureV1?

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case messageType = "message_type"
        case requestID = "request_id"
        case terminalSequence = "terminal_sequence"
        case status
        case finishReason = "finish_reason"
        case completedAtUnixMilliseconds = "completed_at_unix_ms"
        case outputText = "output_text"
        case usage, metrics, execution, failure
    }
}

public enum StrataEnvelopeV1: Equatable, Sendable {
    case start(StrataStartRequestV1)
    case control(StrataControlV1)
    case event(StrataEventV1)
    case receipt(StrataReceiptV1)
}

public enum StrataWireDecoderV1 {
    public static let startMaximumBytes = 1_048_576
    public static let controlMaximumBytes = 16_384
    public static let eventMaximumBytes = 131_072
    public static let receiptMaximumBytes = 2_097_152

    private static let decoder = JSONDecoder()

    public static func decodeEnvelope(_ data: Data) throws -> StrataEnvelopeV1 {
        try JSONDuplicateKeyScanner.scan(data)
        let object = try jsonObject(data)
        guard let type = object["message_type"] as? String else {
            throw StrataWireContractError.invalidValue("message_type")
        }
        switch type {
        case "start": return .start(try decodeStart(data, preflighted: true))
        case "control": return .control(try decodeControl(data, preflighted: true))
        case "event": return .event(try decodeEvent(data, preflighted: true))
        case "receipt": return .receipt(try decodeReceipt(data, preflighted: true))
        default: throw StrataWireContractError.invalidValue("message_type")
        }
    }

    public static func decodeStart(_ data: Data) throws -> StrataStartRequestV1 {
        try decodeStart(data, preflighted: false)
    }

    public static func decodeControl(_ data: Data) throws -> StrataControlV1 {
        try decodeControl(data, preflighted: false)
    }

    public static func decodeEvent(_ data: Data) throws -> StrataEventV1 {
        try decodeEvent(data, preflighted: false)
    }

    public static func decodeReceipt(_ data: Data) throws -> StrataReceiptV1 {
        try decodeReceipt(data, preflighted: false)
    }

    public static func encode<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(value)
    }

    private static func decodeStart(_ data: Data, preflighted: Bool) throws -> StrataStartRequestV1 {
        if !preflighted { try JSONDuplicateKeyScanner.scan(data) }
        try maximum(data, startMaximumBytes, "start")
        let object = try jsonObject(data)
        try exact(object, "start", required: [
            "schema_version", "message_type", "request_id", "task_id", "operation",
            "runtime_revision", "created_at_unix_ms", "deadline_unix_ms", "model",
            "messages", "sampling", "objective", "memory", "deployment_mode",
        ])
        try exact(try child(object, "model"), "model", required: [
            "id", "revision", "artifact_digest", "adapter_id", "quantization",
        ])
        guard let messages = object["messages"] as? [[String: Any]] else {
            throw StrataWireContractError.invalidValue("messages")
        }
        for (index, message) in messages.enumerated() {
            try exact(message, "messages[\(index)]", required: ["role", "content"])
        }
        try exact(try child(object, "sampling"), "sampling", required: ["temperature", "top_p", "seed", "stop"])
        try exact(
            try child(object, "objective"), "objective",
            required: ["workload_class", "context_tokens", "max_output_tokens", "batch_size"],
            optional: ["target_ttft_ms", "min_decode_tokens_per_second"]
        )
        try exact(
            try child(object, "memory"), "memory",
            required: ["max_runtime_memory_bytes", "ssd_offload"],
            optional: ["max_resident_weight_bytes"]
        )
        let request = try decoder.decode(StrataStartRequestV1.self, from: data)
        try validate(request)
        return request
    }

    private static func decodeControl(_ data: Data, preflighted: Bool) throws -> StrataControlV1 {
        if !preflighted { try JSONDuplicateKeyScanner.scan(data) }
        try maximum(data, controlMaximumBytes, "control")
        let object = try jsonObject(data)
        try exact(
            object, "control",
            required: ["schema_version", "message_type", "control_id", "action", "issued_at_unix_ms"],
            optional: ["target_request_id"]
        )
        let control = try decoder.decode(StrataControlV1.self, from: data)
        try validate(control)
        return control
    }

    private static func decodeEvent(_ data: Data, preflighted: Bool) throws -> StrataEventV1 {
        if !preflighted { try JSONDuplicateKeyScanner.scan(data) }
        try maximum(data, eventMaximumBytes, "event")
        let object = try jsonObject(data)
        try exact(
            object, "event",
            required: ["schema_version", "message_type", "request_id", "sequence", "kind", "emitted_at_unix_ms"],
            optional: ["progress", "text_delta", "generated_tokens_delta", "queue_position", "queue_depth", "message"]
        )
        let event = try decoder.decode(StrataEventV1.self, from: data)
        try validate(event)
        return event
    }

    private static func decodeReceipt(_ data: Data, preflighted: Bool) throws -> StrataReceiptV1 {
        if !preflighted { try JSONDuplicateKeyScanner.scan(data) }
        try maximum(data, receiptMaximumBytes, "receipt")
        let object = try jsonObject(data)
        try exact(
            object, "receipt",
            required: [
                "schema_version", "message_type", "request_id", "terminal_sequence",
                "status", "finish_reason", "completed_at_unix_ms", "usage", "metrics", "execution",
            ],
            optional: ["output_text", "failure"]
        )
        try exact(try child(object, "usage"), "usage", required: ["prompt_tokens", "completion_tokens", "total_tokens"])
        try exact(try child(object, "metrics"), "metrics", required: [
            "queue_ms", "ttft_ms", "inter_token_latency_ms", "decode_tokens_per_second", "peak_memory_bytes", "maximum_batch_size",
        ])
        try exact(
            try child(object, "execution"), "execution",
            required: ["backend", "route", "plan_id", "evidence_ids", "fallback_used"],
            optional: ["fallback_reason"]
        )
        if object["failure"] != nil {
            try exact(try child(object, "failure"), "failure", required: ["code", "stage", "retryable", "message"])
        }
        let receipt = try decoder.decode(StrataReceiptV1.self, from: data)
        try validate(receipt)
        return receipt
    }

    private static func validate(_ request: StrataStartRequestV1) throws {
        guard request.schemaVersion == 1, request.messageType == "start",
              validIdentifier(request.requestID), validIdentifier(request.taskID),
              request.operation == "llm.generate",
              request.runtimeRevision.matches("^[0-9a-f]{64}$"),
              request.createdAtUnixMilliseconds >= 0,
              request.deadlineUnixMilliseconds > request.createdAtUnixMilliseconds,
              request.deadlineUnixMilliseconds - request.createdAtUnixMilliseconds <= 86_400_000,
              request.deploymentMode == "supported" else {
            throw StrataWireContractError.invalidValue("start envelope")
        }
        guard validIdentifier(request.model.id),
              request.model.revision.matches("^(?:[0-9a-f]{40}|[0-9a-f]{64})$"),
              request.model.artifactDigest.matches("^sha256:[0-9a-f]{64}$"),
              request.model.adapterID.matches("^[a-z0-9][a-z0-9._-]{0,63}$"),
              (1...64).contains(request.model.quantization.count) else {
            throw StrataWireContractError.invalidValue("model")
        }
        guard (1...128).contains(request.messages.count),
              request.messages.allSatisfy({ (1...65_536).contains($0.content.count) }),
              request.messages.reduce(0, { $0 + $1.content.count }) <= 524_288,
              request.sampling.temperature.isFinite,
              (0...2).contains(request.sampling.temperature),
              request.sampling.topP.isFinite,
              request.sampling.topP > 0, request.sampling.topP <= 1,
              request.sampling.stop.count <= 4,
              Set(request.sampling.stop).count == request.sampling.stop.count,
              request.sampling.stop.allSatisfy({ (1...128).contains($0.count) }) else {
            throw StrataWireContractError.invalidValue("messages or sampling")
        }
        guard (1...131_072).contains(request.objective.contextTokens),
              (1...8_192).contains(request.objective.maxOutputTokens),
              request.objective.batchSize == 1,
              positiveFinite(request.objective.targetTTFTMilliseconds),
              positiveFinite(request.objective.minimumDecodeTokensPerSecond),
              request.memory.maxRuntimeMemoryBytes >= 268_435_456,
              request.memory.maxResidentWeightBytes.map({ $0 > 0 && $0 <= request.memory.maxRuntimeMemoryBytes }) ?? true,
              request.memory.ssdOffload == "disabled" else {
            throw StrataWireContractError.invalidValue("objective or memory")
        }
    }

    private static func validate(_ control: StrataControlV1) throws {
        guard control.schemaVersion == 1, control.messageType == "control",
              validIdentifier(control.controlID), control.issuedAtUnixMilliseconds >= 0 else {
            throw StrataWireContractError.invalidValue("control envelope")
        }
        switch control.action {
        case .cancel:
            guard let target = control.targetRequestID, validIdentifier(target) else {
                throw StrataWireContractError.invalidValue("cancel target")
            }
        case .drain:
            guard control.targetRequestID == nil else {
                throw StrataWireContractError.invalidValue("drain target")
            }
        }
    }

    private static func validate(_ event: StrataEventV1) throws {
        guard event.schemaVersion == 1, event.messageType == "event",
              validIdentifier(event.requestID), event.sequence >= 0,
              event.emittedAtUnixMilliseconds >= 0 else {
            throw StrataWireContractError.invalidValue("event envelope")
        }
        let fields = [
            event.progress != nil, event.textDelta != nil, event.generatedTokensDelta != nil,
            event.queuePosition != nil, event.queueDepth != nil, event.message != nil,
        ]
        switch event.kind {
        case .accepted:
            guard fields[0...4].allSatisfy({ !$0 }) else { throw StrataWireContractError.invalidValue("accepted fields") }
        case .progress:
            guard let progress = event.progress, progress.isFinite, (0...1).contains(progress),
                  event.textDelta == nil, event.generatedTokensDelta == nil,
                  event.queuePosition == nil, event.queueDepth == nil else {
                throw StrataWireContractError.invalidValue("progress fields")
            }
        case .textDelta:
            guard let text = event.textDelta, (1...65_536).contains(text.count),
                  let tokens = event.generatedTokensDelta, (0...8_192).contains(tokens),
                  event.progress == nil, event.queuePosition == nil,
                  event.queueDepth == nil, event.message == nil else {
                throw StrataWireContractError.invalidValue("text delta fields")
            }
        case .queueState:
            guard let position = event.queuePosition, position >= 0,
                  let depth = event.queueDepth, depth >= 0,
                  event.progress == nil, event.textDelta == nil,
                  event.generatedTokensDelta == nil else {
                throw StrataWireContractError.invalidValue("queue fields")
            }
        case .warning:
            guard let message = event.message, (1...1_024).contains(message.count),
                  fields[0...4].allSatisfy({ !$0 }) else {
                throw StrataWireContractError.invalidValue("warning fields")
            }
        }
        if let message = event.message, !(1...1_024).contains(message.count) {
            throw StrataWireContractError.invalidValue("event message")
        }
    }

    private static func validate(_ receipt: StrataReceiptV1) throws {
        guard receipt.schemaVersion == 1, receipt.messageType == "receipt",
              validIdentifier(receipt.requestID), receipt.terminalSequence >= 0,
              receipt.completedAtUnixMilliseconds >= 0,
              receipt.usage.promptTokens >= 0, receipt.usage.completionTokens >= 0,
              receipt.usage.totalTokens == receipt.usage.promptTokens + receipt.usage.completionTokens,
              receipt.metrics.queueMilliseconds.isFinite, receipt.metrics.queueMilliseconds >= 0,
              optionalNonNegativeFinite(receipt.metrics.ttftMilliseconds),
              optionalNonNegativeFinite(receipt.metrics.interTokenLatencyMilliseconds),
              positiveFinite(receipt.metrics.decodeTokensPerSecond),
              receipt.metrics.peakMemoryBytes >= 0, receipt.metrics.maximumBatchSize >= 1,
              validIdentifier(receipt.execution.planID), receipt.execution.evidenceIDs.count <= 16,
              Set(receipt.execution.evidenceIDs).count == receipt.execution.evidenceIDs.count,
              receipt.execution.evidenceIDs.allSatisfy(validIdentifier) else {
            throw StrataWireContractError.invalidValue("receipt fields")
        }
        if receipt.execution.fallbackUsed {
            guard let reason = receipt.execution.fallbackReason, (1...512).contains(reason.count) else {
                throw StrataWireContractError.invalidValue("fallback reason")
            }
        } else if receipt.execution.fallbackReason != nil {
            throw StrataWireContractError.invalidValue("unexpected fallback reason")
        }
        switch (receipt.status, receipt.finishReason) {
        case (.completed, .stop), (.completed, .length):
            guard receipt.outputText != nil, receipt.failure == nil else {
                throw StrataWireContractError.invalidValue("completed receipt")
            }
        case (.failed, .error), (.cancelled, .cancelled), (.rejected, .rejected):
            guard let failure = receipt.failure, validIdentifier(failure.code),
                  (1...1_024).contains(failure.message.count) else {
                throw StrataWireContractError.invalidValue("failure receipt")
            }
            if receipt.status == .rejected && failure.stage != .notStarted {
                throw StrataWireContractError.invalidValue("rejected failure stage")
            }
        default:
            throw StrataWireContractError.invalidValue("status and finish reason")
        }
        if let output = receipt.outputText, output.count > 1_048_576 {
            throw StrataWireContractError.invalidValue("output_text")
        }
    }

    private static func maximum(_ data: Data, _ maximum: Int, _ name: String) throws {
        if data.count > maximum { throw StrataWireContractError.messageTooLarge(name) }
    }

    private static func jsonObject(_ data: Data) throws -> [String: Any] {
        do {
            guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw StrataWireContractError.invalidJSON("top-level value is not an object")
            }
            return object
        } catch let error as StrataWireContractError {
            throw error
        } catch {
            throw StrataWireContractError.invalidJSON(error.localizedDescription)
        }
    }

    private static func child(_ object: [String: Any], _ key: String) throws -> [String: Any] {
        guard let value = object[key] as? [String: Any] else {
            throw StrataWireContractError.invalidValue(key)
        }
        return value
    }

    private static func exact(
        _ object: [String: Any], _ name: String,
        required: Set<String>, optional: Set<String> = []
    ) throws {
        let keys = Set(object.keys)
        let missing = required.subtracting(keys)
        let unknown = keys.subtracting(required.union(optional))
        if !missing.isEmpty || !unknown.isEmpty {
            throw StrataWireContractError.invalidKeys("\(name): missing=\(missing.sorted()), unknown=\(unknown.sorted())")
        }
    }

    private static func validIdentifier(_ value: String) -> Bool {
        value.matches("^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    }

    private static func positiveFinite(_ value: Double?) -> Bool {
        value.map({ $0.isFinite && $0 > 0 }) ?? true
    }

    private static func optionalNonNegativeFinite(_ value: Double?) -> Bool {
        value.map({ $0.isFinite && $0 >= 0 }) ?? true
    }
}

public struct StrataStreamStateV1: Sendable {
    public let requestID: String
    public private(set) var lastSequence: Int64?
    public private(set) var isTerminal = false

    public init(requestID: String) {
        self.requestID = requestID
    }

    public mutating func accept(_ event: StrataEventV1) throws {
        guard !isTerminal, event.requestID == requestID else {
            throw StrataWireContractError.streamViolation("event after terminal or wrong request")
        }
        if let lastSequence {
            guard event.sequence > lastSequence else {
                throw StrataWireContractError.streamViolation("event sequence is not monotonic")
            }
        } else if event.sequence != 0 || event.kind != .accepted {
            throw StrataWireContractError.streamViolation("first event must be accepted sequence 0")
        }
        lastSequence = event.sequence
    }

    public mutating func accept(_ receipt: StrataReceiptV1) throws {
        guard !isTerminal, receipt.requestID == requestID else {
            throw StrataWireContractError.streamViolation("duplicate terminal or wrong request")
        }
        if let lastSequence, receipt.terminalSequence <= lastSequence {
            throw StrataWireContractError.streamViolation("terminal sequence is not monotonic")
        }
        lastSequence = receipt.terminalSequence
        isTerminal = true
    }
}

private extension String {
    func matches(_ pattern: String) -> Bool {
        range(of: pattern, options: .regularExpression) != nil
    }
}

private enum JSONDuplicateKeyScanner {
    static func scan(_ data: Data) throws {
        var parser = Parser(bytes: Array(data))
        try parser.parseValue()
        parser.skipWhitespace()
        guard parser.index == parser.bytes.count else {
            throw StrataWireContractError.invalidJSON("trailing bytes")
        }
    }

    private struct Parser {
        let bytes: [UInt8]
        var index = 0

        mutating func skipWhitespace() {
            while index < bytes.count && [9, 10, 13, 32].contains(bytes[index]) { index += 1 }
        }

        mutating func parseValue() throws {
            skipWhitespace()
            guard index < bytes.count else { throw invalid("unexpected end") }
            switch bytes[index] {
            case 123: try parseObject()
            case 91: try parseArray()
            case 34: _ = try parseString()
            default: try parsePrimitive()
            }
        }

        mutating func parseObject() throws {
            index += 1
            skipWhitespace()
            var keys = Set<String>()
            if consume(125) { return }
            while true {
                skipWhitespace()
                let key = try parseString()
                if !keys.insert(key).inserted { throw StrataWireContractError.duplicateKey(key) }
                skipWhitespace()
                guard consume(58) else { throw invalid("missing object colon") }
                try parseValue()
                skipWhitespace()
                if consume(125) { return }
                guard consume(44) else { throw invalid("missing object comma") }
            }
        }

        mutating func parseArray() throws {
            index += 1
            skipWhitespace()
            if consume(93) { return }
            while true {
                try parseValue()
                skipWhitespace()
                if consume(93) { return }
                guard consume(44) else { throw invalid("missing array comma") }
            }
        }

        mutating func parseString() throws -> String {
            guard consume(34) else { throw invalid("expected string") }
            let start = index - 1
            var escaped = false
            while index < bytes.count {
                let byte = bytes[index]
                index += 1
                if escaped { escaped = false; continue }
                if byte == 92 { escaped = true; continue }
                if byte == 34 {
                    let encoded = Data(bytes[start..<index])
                    do { return try JSONDecoder().decode(String.self, from: encoded) }
                    catch { throw invalid("invalid string") }
                }
            }
            throw invalid("unterminated string")
        }

        mutating func parsePrimitive() throws {
            let start = index
            while index < bytes.count && ![9, 10, 13, 32, 44, 93, 125].contains(bytes[index]) {
                index += 1
            }
            guard index > start else { throw invalid("invalid primitive") }
        }

        mutating func consume(_ byte: UInt8) -> Bool {
            guard index < bytes.count, bytes[index] == byte else { return false }
            index += 1
            return true
        }

        func invalid(_ message: String) -> StrataWireContractError {
            .invalidJSON(message)
        }
    }
}
