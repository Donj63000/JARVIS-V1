# JARVIS V1 (LocalAI)

Local desktop chatbot in Rust with:
- Windows 95 / MSN-like chat window (`eframe`/`egui`)
- Local model backend via Ollama HTTP API
- Grounded answers on the Rochias production guide (RAG-style retrieval)
- Citation-first responses (`[section|Lx-Ly]`) and safe abstention when unknown
- CLI fallback mode
- Unit and integration tests

## Quick start (project-local Ollama via Docker)

1. Start Ollama in Docker and pull a model into the project volume:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-local-model.ps1 -Model qwen2.5-coder:14b
```

2. Launch the GUI chat app:

```powershell
cargo run --release
```

If `guide-production-rochias.txt` is present at project root, guide-grounded mode is enabled automatically.

3. Stop local Ollama service when done:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stop-local-model.ps1
```

## CLI mode

One-shot:

```powershell
cargo run --release -- --cli "Explain Rust ownership in 4 bullets."
```

Streaming:

```powershell
cargo run --release -- --cli --stream "Write a Rust function that sorts Vec<i32>."
```

Grounded CLI with guide:

```powershell
cargo run --release -- --cli --model qwen2.5-coder:14b --guide guide-production-rochias.txt "Quel signal est mentionne en cas de defaut ?"
```

## Config options

All modes support:
- `--host` (default `http://localhost:11434`)
- `--model` (default `qwen2.5-coder:14b`)
- `--system`
- `--temperature`
- `--max-tokens`
- `--timeout-seconds` (default `600`)
- `--guide` (default `guide-production-rochias.txt`)
- `--no-guide` (disable grounded retrieval)
- `--source-limit` (default `6`)

## Guide-grounded behavior

- The app chunks and indexes the production guide.
- Each question retrieves top relevant passages.
- The model is prompted to answer only from these passages.
- If evidence is missing, it must answer: `Information non trouvee dans le guide fourni.`
- In GUI, model selector includes: `qwen2.5-coder:14b`, `gpt-oss:20b`.
- When `gpt-oss:20b` is selected, a reasoning mode control appears (`Bas`, `Moyen`, `Haut`) and the right panel shows `Chaine de pensee`.

## Tests

Run all tests:

```powershell
cargo test
```

Run guide RAG evaluation against the local model:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\eval-guide-rag.ps1 -Model qwen2.5-coder:14b
```

Quick smoke run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\eval-guide-rag.ps1 -Model qwen2.5-coder:14b -MaxQuestions 3
```

What is tested:
- NDJSON stream parsing
- OpenAI-compatible response extraction
- HTTP integration against mock server for both `/v1/chat/completions` and `/api/chat`
- Guide retrieval quality on real guide facts (when file is present)
- Guardrail/citation prompt structure

## Project files

- `src/chat_api.rs`: backend client and parsers
- `src/guide_knowledge.rs`: guide indexing, retrieval, grounding prompts
- `src/gui.rs`: messenger-style GUI
- `src/main.rs`: entrypoint (GUI default, CLI optional)
- `docker-compose.yml`: local Ollama service
- `scripts/start-local-model.ps1`: start service + pull model
- `scripts/eval-guide-rag.ps1`: automated grounded-answer check
