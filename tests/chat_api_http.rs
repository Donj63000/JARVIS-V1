use httpmock::prelude::*;
use localai::chat_api::{ChatClient, ChatConfig};
use serde_json::json;

#[tokio::test]
async fn oneshot_returns_first_choice_content() {
    let server = MockServer::start();
    let expected = "Salut depuis le mock";

    let mock = server.mock(|when, then| {
        when.method(POST).path("/v1/chat/completions");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "choices": [
                    { "message": { "content": expected } }
                ]
            }));
    });

    let config = ChatConfig {
        host: server.base_url(),
        ..ChatConfig::default()
    };
    let client = ChatClient::new(config).unwrap();

    let answer = client.oneshot("Bonjour ?").await.unwrap();

    mock.assert();
    assert_eq!(answer, expected);
}

#[tokio::test]
async fn stream_concatenates_ndjson_chunks() {
    let server = MockServer::start();
    let ndjson = concat!(
        "{\"message\":{\"content\":\"Bon\"},\"done\":false}\n",
        "{\"message\":{\"content\":\"jour\"},\"done\":false}\n",
        "{\"done\":true}\n"
    );

    let mock = server.mock(|when, then| {
        when.method(POST).path("/api/chat");
        then.status(200)
            .header("content-type", "application/x-ndjson")
            .body(ndjson);
    });

    let config = ChatConfig {
        host: server.base_url(),
        ..ChatConfig::default()
    };
    let client = ChatClient::new(config).unwrap();

    let mut answer = String::new();
    client
        .stream("Hello", |chunk| {
            answer.push_str(chunk);
        })
        .await
        .unwrap();

    mock.assert();
    assert_eq!(answer, "Bonjour");
}

#[tokio::test]
async fn stream_uses_num_predict_from_config() {
    let server = MockServer::start();

    let mock = server.mock(|when, then| {
        when.method(POST)
            .path("/api/chat")
            .body_contains("\"num_predict\":777");
        then.status(200)
            .header("content-type", "application/x-ndjson")
            .body("{\"done\":true}\n");
    });

    let config = ChatConfig {
        host: server.base_url(),
        max_tokens: 777,
        ..ChatConfig::default()
    };
    let client = ChatClient::new(config).unwrap();

    client.stream("Test", |_chunk| {}).await.unwrap();

    mock.assert();
}
