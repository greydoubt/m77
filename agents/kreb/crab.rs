use agentic_llm::{LLMAdapter, OpenAIAdapter, LLMMessage};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let adapter = OpenAIAdapter::new("sk-...", "gpt-4o");

    let messages = vec![
        LLMMessage::system("You are a helpful assistant."),
        LLMMessage::user("Hello!"),
    ];

    let response = adapter.generate(&messages).await?;
    println!("{}", response.content);

    Ok(())
}
