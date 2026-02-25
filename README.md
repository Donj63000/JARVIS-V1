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

## Configuration minimale recommandee (ca marche, mais ca peut ramer)

Ces valeurs sont des minimums pratiques pour lancer l'app avec les modeles du projet.
En dessous, ca peut ne pas demarrer ou devenir tres lent (swap disque, reponses longues, freeze UI).

- OS:
  - Windows 10 22H2+ (ou Linux/macOS avec Docker/Ollama)
- RAM systeme:
  - minimum 16 GB pour `qwen2.5-coder:14b` ou `gpt-oss:20b`
  - a 16 GB, attendez-vous a des ralentissements en prompts longs et en mode raisonnement `Haut`
- GPU (optionnel mais conseille):
  - minimum 12 GB VRAM pour `qwen2.5-coder:14b`
  - minimum 16 GB VRAM pour `gpt-oss:20b`
  - sans GPU, execution CPU possible mais plus lente
- Stockage libre:
  - minimum 25 GB si vous gardez un seul modele (app + Ollama + marge)
  - minimum 40 GB si vous gardez `qwen2.5-coder:14b` et `gpt-oss:20b`

Recommande pour un usage fluide:
- 32 GB RAM
- GPU 16 GB VRAM+
- SSD NVMe

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

Sources techniques (tailles/memoire):
- https://ollama.com/library/qwen2.5-coder:14b
- https://ollama.com/library/gpt-oss:20b
- https://docs.ollama.com/windows

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

## No dev (pas a pas Windows)

Cette section explique comment utiliser le projet meme si vous n'etes pas developpeur.

### 1) Installer les prerequis (une seule fois)

- Git for Windows: https://git-scm.com/download/win
- Rust (rustup): https://www.rust-lang.org/tools/install
- Visual Studio Build Tools (C++): https://visualstudio.microsoft.com/fr/visual-cpp-build-tools/
- Docker Desktop: https://www.docker.com/products/docker-desktop/

Important:
- Activez WSL2 dans Docker Desktop si demande.
- Redemarrez le PC apres l'installation des outils si Windows le demande.

### 2) Telecharger le projet

Ouvrez PowerShell puis lancez:

```powershell
cd $HOME
git clone https://github.com/Donj63000/JARVIS-V1.git
cd .\JARVIS-V1
```

### 3) Telecharger un modele IA en local

Exemple avec `qwen2.5-coder:14b`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-local-model.ps1 -Model qwen2.5-coder:14b
```

Exemple avec `gpt-oss:20b`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-local-model.ps1 -Model gpt-oss:20b
```

Notes:
- Le premier telechargement peut etre long (plusieurs Go).
- Les modeles ne sont pas dans ce repo: chacun les telecharge sur son PC.

### 4) Compiler l'application

```powershell
cargo build --release
```

La premiere compilation peut prendre plusieurs minutes.

### 5) Lancer le logiciel

```powershell
.\target\release\LocalAI.exe
```

Alternative:

```powershell
cargo run --release
```

### 6) Utiliser l'application

- Verifier `Host`: `http://localhost:11434`
- Choisir le modele dans le selecteur
- Ecrire un message puis cliquer sur `Envoyer`

### 7) Arreter le backend local quand vous avez fini

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stop-local-model.ps1
```

### Depannage rapide

- Erreur script PowerShell: lancez une fois `Set-ExecutionPolicy -Scope Process Bypass`
- Erreur Docker/Ollama: verifiez que Docker Desktop est bien demarre
- Reponse tres lente: prenez un modele plus leger ou baissez le niveau de raisonnement

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
