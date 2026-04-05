<div align="center">

# M-flow

**A memory engine that focuses on reasoning and association.**

M-flow 是一款重构检索架构、侧重推理与联想的记忆引擎。

[Quick Start](#quick-start) ·
[Architecture](docs/RETRIEVAL_ARCHITECTURE.md) ·
[架构详解（中文）](docs/RETRIEVAL_ARCHITECTURE_ZH.md) ·
[Examples](examples/) ·
[Contact](mailto:contact@xinliuyuansu.com)

[![Tests](https://img.shields.io/badge/tests-963%20passed-brightgreen)](#testing)
[![Python](https://img.shields.io/badge/python-3.10–3.13-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

</div>

---

## What is M-flow?

M-flow is not a vector database or a RAG framework. It is a **cognitive memory engine** — a system that doesn't just store text, but *understands* it, distilling incoming knowledge into structured layers and weaving them into the right place of a persistent architecture.

Where traditional retrieval matches queries by surface similarity, M-flow **reasons through relationships** — tracing connections, weighing context, and reaching answers the way cognition does.

Most AI memory systems fall into two categories:

- **Context Window** — The model re-reads everything from scratch. Exhaustive, linear, forgetful.
- **Similarity Search** — External memory exists, but retrieval is surface-level — matching by vector shape, blind to structure. Approximate, lossy.

M-flow introduces a third paradigm: a **Cognitive Engine** where memory becomes a mind. A query at any granularity finds a precise anchor, then expands outward — surfacing related context and associative content across different levels of detail.

## How It Works

M-flow organizes knowledge into a four-level **Cone Graph** — a layered hierarchy from abstract summaries to atomic facts:

| Level | What it captures | Query example |
|-------|-----------------|---------------|
| **Episode** | A bounded semantic focus — an incident, decision process, or workflow | *"What happened with the tech stack decision?"* |
| **Facet** | One dimension of that Episode — a topical cross-section | *"What were the performance targets?"* |
| **FacetPoint** | An atomic assertion or fact derived from a Facet | *"Was the P99 target under 500ms?"* |
| **Entity** | A named thing — person, tool, metric — linked across all Episodes | *"Tell me about GPT-4o"* → surfaces all related contexts |

Retrieval is **graph-routed**: the system casts a wide net across all levels, projects hits into the knowledge graph, propagates cost along every possible path, and scores each Episode by its **tightest chain of evidence**. One strong path is enough — the way a single association triggers an entire memory.

> For the full technical deep-dive, see [Retrieval Architecture](docs/RETRIEVAL_ARCHITECTURE.md) | [检索架构详解（中文）](docs/RETRIEVAL_ARCHITECTURE_ZH.md)

## Benchmark

Dataset: [LoCoMo-10](https://github.com/snap-stanford/locomo) (10 conversations, 1540 questions) · Judge LLM: gpt-4o-mini

| System | LLM-Judge | BLEU-1 | F1 |
|--------|-----------|--------|-----|
| **M-flow** | **76.5%** | **0.422** | **0.503** |
| Mem0 Cloud (tested) | 40.4% | 0.186 | 0.196 |

| Category | M-flow | Mem0 Cloud (tested) |
|----------|--------|---------------------|
| Single-hop (factual) | 68.5% | 44.7% |
| Temporal reasoning | **78.8%** | 7.5% |
| Multi-hop reasoning | 48.0% | 39.6% |
| Open-domain (event detail) | **81.5%** | 49.1% |

## Features

| | |
|---------|-------------|
| **Episodic + Procedural memory** | Hierarchical recall for facts and step-by-step knowledge |
| **5 retrieval modes** | Episodic, Procedural, Triplet Completion, Lexical, Cypher |
| **50+ file formats** | PDFs, DOCX, HTML, Markdown, images, audio, and more |
| **Multi-DB support** | LanceDB, Neo4j, PostgreSQL/pgvector, ChromaDB, KùzuDB, Pinecone |
| **LLM-agnostic** | OpenAI, Anthropic, Mistral, Groq, Ollama, LLaMA-Index, LangChain |
| **MCP server** | Expose memory as Model Context Protocol tools for any IDE |
| **CLI & Web UI** | Interactive console, knowledge graph visualization, config wizard |

## Quick Start

### One-Command Setup (Docker)

```bash
git clone https://github.com/FlowElement-ai/m_flow.git && cd m_flow
./quickstart.sh
```

The script checks your environment, configures API keys interactively, and starts the full stack (backend + frontend). On Windows, use `.\quickstart.ps1`.

### Install via pip

```bash
pip install mflow-ai         # or: uv pip install mflow-ai
export LLM_API_KEY="sk-..."
```

### Install from Source

```bash
git clone https://github.com/FlowElement-ai/m_flow.git && cd m_flow
pip install -e .             # editable install for development
```

### Run

```python
import asyncio
import m_flow


async def main():
    await m_flow.add("M-flow builds persistent memory for AI agents.")
    await m_flow.memorize()

    results = await m_flow.search("How does M-flow work?")
    for r in results:
        print(r)


asyncio.run(main())
```

### CLI

```bash
mflow add "M-flow builds persistent memory for AI agents."
mflow memorize
mflow search "How does M-flow work?"
mflow -ui          # Launch the local web console
```

## Architecture Overview

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  Data Input   │────▶│    Extract    │────▶│   Memorize    │
│  (50+ formats)│     │  (chunking,   │     │  (KG build,   │
│               │     │   parsing)    │     │  embeddings)  │
└───────────────┘     └───────────────┘     └───────┬───────┘
                                                    │
                      ┌───────────────┐     ┌───────▼───────┐
                      │    Search     │◀────│     Load      │
                      │  (graph-routed│     │   (graph +    │
                      │  bundle search│     │  vector DB)   │
                      └───────────────┘     └───────────────┘
```

## Project Layout

```
m_flow/              Core Python library & API
├── api/             FastAPI routers (add, memorize, search, …)
├── cli/             Command-line interface (`mflow`)
├── adapters/        DB adapters (graph, vector, cache)
├── core/            Domain models (Episode, Facet, FacetPoint, …)
├── memory/          Memory processing (episodic, procedural)
├── retrieval/       Search & retrieval algorithms
├── pipeline/        Composable pipeline tasks & orchestration
├── auth/            Authentication & multi-tenancy
├── shared/          Logging, settings, cross-cutting utilities
└── tests/           Unit & integration tests

m_flow-frontend/     Next.js web console
m_flow-mcp/          Model Context Protocol server
mflow_workers/       Distributed execution helpers (Modal, workers)
examples/            Runnable example scripts
docs/                Architecture & design documents
```

## Development

```bash
git clone https://github.com/FlowElement-ai/m_flow.git && cd m_flow
uv sync --dev --all-extras --reinstall

# Test
PYTHONPATH=. uv run pytest m_flow/tests/unit/ -v

# Lint
uv run ruff check . && uv run ruff format .
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full contributor guide.

## Deployment

### Docker

```bash
docker compose up                       # Backend only
docker compose --profile ui up          # Backend + frontend
docker compose --profile neo4j up       # Backend + Neo4j
```

### MCP Server

```bash
cd m_flow-mcp
uv sync --dev --all-extras
uv run python src/server.py --transport sse
```

## Testing

```bash
PYTHONPATH=. pytest m_flow/tests/unit/ -v        # ~963 test cases
PYTHONPATH=. pytest m_flow/tests/integration/ -v  # Needs .env with API keys
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines, and our [Code of Conduct](CODE_OF_CONDUCT.md) for community standards.

## License

M-flow is licensed under the [Apache License 2.0](LICENSE).

```
Copyright 2026 Junting Hua

Licensed under the Apache License, Version 2.0.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
```
