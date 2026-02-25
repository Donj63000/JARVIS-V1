# JARVIS V1 (LocalAI)

Concept d'assistant IA 100% local en Rust:
- interface desktop style Messenger 95 (`eframe`/`egui`)
- backend local via API HTTP Ollama
- mode guide (RAG) base sur `guide-production-rochias.txt`
- mode conversation libre
- tests unitaires et d'integration

## Philosophie du projet

L'objectif est simple: faire tourner une IA sur la machine locale, sans dependre d'un service cloud pour les usages standards.

Les donnees de conversation, le guide metier et l'inference restent en local (selon votre configuration Docker/Ollama).

## Modeles utilises et credits (important)

Ce projet utilise des modeles open-weight distribues via Ollama.
Nous ne revendiquons pas la propriete de ces modeles. Le credit revient integralement a leurs createurs.

- `qwen2.5-coder:14b`
  - Createur: Qwen Team (Alibaba Cloud)
  - Reference: https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct
- `gpt-oss:20b`
  - Createur: OpenAI
  - References:
    - https://openai.com/index/introducing-gpt-oss
    - https://openai.com/index/gpt-oss-model-card
    - https://ollama.com/library/gpt-oss:20b

Respectez toujours la licence et les conditions d'usage de chaque modele.

## Demarrage rapide (Ollama local via Docker)

1. Demarrer Ollama et telecharger un modele dans le volume du projet:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-local-model.ps1 -Model qwen2.5-coder:14b
```

2. Lancer l'application GUI:

```powershell
cargo run --release
```

Si `guide-production-rochias.txt` est present a la racine, le mode guide est active automatiquement.

3. Arreter Ollama local:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stop-local-model.ps1
```

## Mode CLI

Commande simple:

```powershell
cargo run --release -- --cli "Explique ownership en Rust en 4 points."
```

Streaming:

```powershell
cargo run --release -- --cli --stream "Ecris une fonction Rust qui trie un Vec<i32>."
```

CLI avec guide:

```powershell
cargo run --release -- --cli --model qwen2.5-coder:14b --guide guide-production-rochias.txt "Quel signal est mentionne en cas de defaut ?"
```

## Options de configuration

Disponibles en GUI/CLI:
- `--host` (defaut `http://localhost:11434`)
- `--model` (defaut `qwen2.5-coder:14b`)
- `--system`
- `--temperature`
- `--max-tokens`
- `--timeout-seconds` (defaut `600`)
- `--guide` (defaut `guide-production-rochias.txt`)
- `--no-guide` (desactive le RAG guide)
- `--source-limit` (defaut `6`)

## Comportement du mode guide (RAG)

- Le guide est segmente, indexe puis interroge.
- Chaque question recupere les passages les plus pertinents.
- Le modele est contraint a repondre uniquement avec ces passages.
- Si aucune preuve n'est trouvee: `Information non trouvee dans le guide fourni.`
- Dans l'UI, les modeles proposes sont `qwen2.5-coder:14b` et `gpt-oss:20b`.
- Avec `gpt-oss:20b`, le choix du niveau de raisonnement (`Bas`, `Moyen`, `Haut`) est disponible et la `Chaine de pensee` est affichee a droite.

## Tests

Tout lancer:

```powershell
cargo test
```

Evaluation guide RAG sur modele local:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\eval-guide-rag.ps1 -Model qwen2.5-coder:14b
```

Smoke test rapide:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\eval-guide-rag.ps1 -Model qwen2.5-coder:14b -MaxQuestions 3
```

Ce qui est teste:
- parsing NDJSON streaming
- extraction de reponses OpenAI-compatibles
- integration HTTP mockee (`/v1/chat/completions` et `/api/chat`)
- qualite de retrieval du guide (si le fichier existe)
- structure des prompts (garde-fous/citations)

## Fichiers principaux

- `src/chat_api.rs`: client backend + parseurs
- `src/guide_knowledge.rs`: indexation/retrieval/grounding du guide
- `src/gui.rs`: interface style messenger
- `src/main.rs`: point d'entree (GUI par defaut, CLI possible)
- `docker-compose.yml`: service Ollama local
- `scripts/start-local-model.ps1`: demarrage + pull modele
- `scripts/eval-guide-rag.ps1`: verification auto des reponses guidees
