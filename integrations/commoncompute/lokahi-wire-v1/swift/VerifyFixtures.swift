import Foundation

@main
enum VerifyFixtures {
    static func main() throws {
        guard CommandLine.arguments.count == 2 else {
            throw LokahiWireContractError.invalidValue("expected kit root argument")
        }
        let root = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
        let manager = FileManager.default
        var validCount = 0
        var invalidCount = 0

        let valid = try fixtureURLs(root.appendingPathComponent("fixtures/valid"), manager)
        for url in valid {
            let data = try Data(contentsOf: url)
            let envelope = try LokahiWireDecoderV1.decodeEnvelope(data)
            let encoded: Data
            switch envelope {
            case .start(let value): encoded = try LokahiWireDecoderV1.encode(value)
            case .control(let value): encoded = try LokahiWireDecoderV1.encode(value)
            case .event(let value): encoded = try LokahiWireDecoderV1.encode(value)
            case .receipt(let value): encoded = try LokahiWireDecoderV1.encode(value)
            }
            let originalObject = try JSONSerialization.jsonObject(with: data) as! NSDictionary
            let encodedObject = try JSONSerialization.jsonObject(with: encoded) as! NSDictionary
            guard originalObject == encodedObject else {
                throw LokahiWireContractError.invalidValue("round-trip mismatch: \(url.lastPathComponent)")
            }
            validCount += 1
        }

        let invalid = try fixtureURLs(root.appendingPathComponent("fixtures/invalid"), manager)
        for url in invalid {
            do {
                _ = try LokahiWireDecoderV1.decodeEnvelope(Data(contentsOf: url))
                throw LokahiWireContractError.invalidValue("invalid fixture passed: \(url.lastPathComponent)")
            } catch LokahiWireContractError.invalidValue(let message)
                where message.hasPrefix("invalid fixture passed:") {
                throw LokahiWireContractError.invalidValue(message)
            } catch {
                invalidCount += 1
            }
        }

        print("{\"invalid\":\(invalidCount),\"success\":true,\"valid\":\(validCount)}")
    }

    private static func fixtureURLs(_ directory: URL, _ manager: FileManager) throws -> [URL] {
        try manager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "json" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
    }
}
