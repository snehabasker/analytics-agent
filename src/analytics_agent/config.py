"""Runtime configuration from environment variables (12-factor style)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    db_path: Path = field(default_factory=lambda: Path(os.getenv("WAREHOUSE_PATH", ROOT / "data" / "warehouse.duckdb")))
    # "claude" needs ANTHROPIC_API_KEY; "heuristic" is the free, deterministic offline planner;
    # "dspy" is the DSPy planner on any LiteLLM model (default: local Ollama), optionally with an optimised program.
    llm: str = field(default_factory=lambda: os.getenv("ANALYTICS_LLM", "auto"))
    model: str = field(default_factory=lambda: os.getenv("ANALYTICS_MODEL", "claude-sonnet-5-5"))
    # Narration is summarisation of already-computed rows: a smaller, cheaper model is enough.
    narrate_model: str = field(default_factory=lambda: os.getenv("ANALYTICS_NARRATE_MODEL", "claude-haiku-5-5"))
    qdrant_url: str | None = field(default_factory=lambda: os.getenv("QDRANT_URL") or None)
    # Retrieval backends. "hashing"/"none" are offline; "cohere" needs COHERE_API_KEY + `pip install cohere`.
    embedder: str = field(default_factory=lambda: os.getenv("ANALYTICS_EMBEDDER", "hashing"))
    reranker: str = field(default_factory=lambda: os.getenv("ANALYTICS_RERANKER", "none"))
    cohere_embed_model: str = field(default_factory=lambda: os.getenv("COHERE_EMBED_MODEL", "embed-v4.0"))
    cohere_rerank_model: str = field(default_factory=lambda: os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5"))
    dspy_lm: str = field(default_factory=lambda: os.getenv("ANALYTICS_DSPY_LM", "ollama_chat/qwen2.5:7b"))
    dspy_program: Path | None = field(default_factory=lambda: Path(p) if (p := os.getenv("ANALYTICS_DSPY_PROGRAM")) else None)
    trace_file: Path | None = field(default_factory=lambda: Path(p) if (p := os.getenv("TRACE_FILE")) else None)
    max_rows: int = field(default_factory=lambda: int(os.getenv("MAX_ROWS", "200")))
    query_timeout_s: float = field(default_factory=lambda: float(os.getenv("QUERY_TIMEOUT_S", "10")))
    max_repair_attempts: int = field(default_factory=lambda: int(os.getenv("MAX_REPAIR_ATTEMPTS", "2")))

    def resolved_llm(self) -> str:
        if self.llm != "auto":
            return self.llm
        return "claude" if os.getenv("ANTHROPIC_API_KEY") else "heuristic"
