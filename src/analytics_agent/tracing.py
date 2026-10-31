"""Dependency-free tracing: nested spans per request, exported as JSONL.

The span schema (trace_id, span_id, parent_id, name, start, duration_ms, attrs) is
OpenTelemetry-shaped so swapping in an OTLP exporter later is a small change.
"""
from __future__ import annotations

import contextvars
import json
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

_current: contextvars.ContextVar["Trace | None"] = contextvars.ContextVar("trace", default=None)
_stack: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar("span_stack", default=())
_write_lock = threading.Lock()


class Span:
    def __init__(self, name: str, parent_id: str | None):
        self.span_id = uuid.uuid4().hex[:16]
        self.parent_id = parent_id
        self.name = name
        self.start = time.time()
        self.duration_ms = 0.0
        self.attrs: dict[str, Any] = {}
        self.error: str | None = None

    def set(self, **attrs: Any) -> None:
        self.attrs.update(attrs)

    def to_dict(self, trace_id: str) -> dict[str, Any]:
        return {"trace_id": trace_id, "span_id": self.span_id, "parent_id": self.parent_id, "name": self.name,
                "start": self.start, "duration_ms": round(self.duration_ms, 2), "attrs": self.attrs, "error": self.error}


class Trace:
    def __init__(self, sink: Path | None = None):
        self.trace_id = uuid.uuid4().hex
        self.spans: list[Span] = []
        self._sink = sink

    def export(self) -> list[dict[str, Any]]:
        return [s.to_dict(self.trace_id) for s in self.spans]

    def flush(self) -> None:
        if self._sink:
            with _write_lock, self._sink.open("a", encoding="utf-8") as f:
                for s in self.export():
                    f.write(json.dumps(s, default=str) + "\n")


@contextmanager
def start_trace(sink: Path | None = None) -> Iterator[Trace]:
    trace = Trace(sink)
    token = _current.set(trace)
    stack_token = _stack.set(())
    try:
        yield trace
    finally:
        _current.reset(token)
        _stack.reset(stack_token)
        trace.flush()


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[Span]:
    """Record a span on the active trace; a harmless no-op recorder if none is active."""
    trace = _current.get()
    stack = _stack.get()
    s = Span(name, stack[-1] if stack else None)
    s.attrs.update(attrs)
    token = _stack.set(stack + (s.span_id,))
    t0 = time.perf_counter()
    try:
        yield s
    except Exception as e:
        s.error = f"{type(e).__name__}: {e}"
        raise
    finally:
        s.duration_ms = (time.perf_counter() - t0) * 1000
        _stack.reset(token)
        if trace is not None:
            trace.spans.append(s)
