import FoundationModels

let model = SystemLanguageModel.default
guard model.availability == .available else { return }

let session = LanguageModelSession {
  """
  Summarise the following:
The matter power spectrum, denoted as P(k)P(k), quantifies the statistical distribution of density fluctuations in the universe as a function of scale, characterized by the wavenumber kk. It is defined as the Fourier transform of the two-point correlation function of the matter density contrast

  """
}

let response = try await session.respond(options: .init(maximumResponseTokens: 1_000)) {
  articleText
}

let markdown = response.content
