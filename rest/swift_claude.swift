import FoundationModels
import ClaudeForFoundationModels

let model = ClaudeLanguageModel(
  name: .sonnet4_6,
  auth: .apiKey(ProcessInfo.processInfo.environment["ANTHROPIC_API_KEY"] ?? "")
  //
  //auth: .proxied(headers: ["X-App-Token": "..."]),
  //baseURL: URL(string: "https://api.yourapp.com/claude")!

  //
  //
  //  serverTools: [
  //  .webSearch(maxUses: 5),
  //  .codeExecution,
)

let session = LanguageModelSession(model: model)
let response = try await session.respond(to: "Plan a 4-day trip to Buenos Aires.")
print(response.content)


let model = ClaudeModel(
  id: "claude-experimental-x",
  capabilities: .init(samplingParams: false, effortLevels: [.low, .high])
  //ClaudeLanguageModel(name: .opus4_8, auth: auth, fixedEffort: .xhigh) Pin a Claude effort level for every request with fixedEffort:. It takes precedence over the framework's per-request reasoning hints, and it's the only way to request .xhigh or .max, because the framework's reasoning levels stop at high. The API defaults to high when no effort is sent
)
ClaudeLanguageModel(name: model, auth: auth)


let stream = session.streamResponse(to: "Summarize today's top science stories.")
for try await partial in stream {
  print(partial.content)
}



@Generable
struct Trip {
  @Guide(description: "Destination city") var destination: String
  @Guide(description: "Length in days") var days: Int
}

let response = try await session.respond(to: "Plan a trip to Tokyo.", generating: Trip.self)
print(response.content.destination)



let session = LanguageModelSession(model: model, tools: [SetCoverTool()])


do {
  let response = try await session.respond(to: prompt)
  print(response.content)
} catch ClaudeError.missingCredential {
  // Prompt for an API key.
} catch let error as LanguageModelError {
  // Framework-shaped errors (rate limits, guardrails, context length, decoding).
} catch {
  // Transport errors.
}
