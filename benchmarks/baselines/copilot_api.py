"""Shared authenticated transport and snapshot validation for Copilot baselines."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import PurePosixPath
from typing import Any

from benchmarks.data import RepoSpec

_MODEL = "metis-1024-I16-Binary"
_TOP_K = 10
_JSON = dict[str, Any]


class APIError(RuntimeError):
    """An HTTP failure, distinct from a successful empty retrieval."""

    def __init__(self, status: int, message: str, headers: dict[str, str] | None = None) -> None:
        """Retain status and retry metadata, without credentials or request headers."""
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.headers = headers or {}


class ProtocolError(ValueError):
    """The index or response cannot be attributed to the requested snapshot."""


class CopilotAPI:
    """Use gh's HTTP stack, bounded safe retries and shared rate-limit guards."""

    def __init__(self, token: str, request_interval: float = 0.5) -> None:
        """Initialize the transport and cross-worker admission/rate-limit guards."""
        self.token = token
        self.request_interval = request_interval
        self.search_limited = threading.Event()
        self.index_limited = threading.Event()
        self.ingest_limited = threading.Event()
        self._schedule_lock = threading.Lock()
        self._next_search = 0.0

    def _once(self, method: str, path: str, body: _JSON | None) -> _JSON:
        command = ["gh", "api", "--hostname", "github.com", "--method", method,
                   "--include", "-H", "Accept: application/json", path]
        if body is not None:
            command.extend(["--input", "-"])
        process = subprocess.run(
            command, input=json.dumps(body) if body is not None else None,
            capture_output=True, text=True, timeout=90, check=False,
            env={**os.environ, "GH_TOKEN": self.token, "GH_DEBUG": "", "GH_PAGER": "cat"},
        )
        header, separator, raw = process.stdout.partition("\n\n")
        if not separator or not header.startswith("HTTP/"):
            raise ConnectionError(f"gh transport failed: {process.stderr[-400:]}")
        status = int(header.splitlines()[0].split()[1])
        headers = {}
        for line in header.splitlines()[1:]:
            key, separator, value = line.partition(":")
            if separator:
                headers[key.lower()] = value.strip()
        try:
            payload = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError as error:
            raise ConnectionError(f"Invalid JSON response (HTTP {status})") from error
        if not isinstance(payload, dict):
            raise ProtocolError("Expected a JSON object")
        if not 200 <= status < 300:
            raise APIError(status, str(payload.get("message", "request failed")), headers)
        if process.returncode:
            raise ConnectionError(f"gh failed after HTTP {status}")
        return payload

    def request(self, method: str, path: str, body: _JSON | None = None) -> _JSON:
        """Request JSON; do not retry index triggers or automatically bypass rate limits."""
        is_search = method == "POST" and path in ("/embeddings/code/search", "/external/embeddings/code/search")
        is_index = method == "POST" and path.endswith("/copilot_internal/embeddings_index")
        is_ingest = method == "POST" and path.startswith("/external/code/ingest")
        if ((is_search and self.search_limited.is_set()) or (is_index and self.index_limited.is_set())
                or (is_ingest and self.ingest_limited.is_set())):
            raise APIError(429, "Further requests paused after a rate limit; resume later")
        # A document upload is idempotent for the same ingest_id/doc_id/content.
        # Do not apply this to snapshot creation or index triggers.
        is_document = method == "POST" and path == "/external/code/ingest/document"
        attempts = 3 if method == "GET" or is_search or is_document else 1
        for attempt in range(attempts):
            try:
                return self._once(method, path, body)
            except APIError as error:
                self._mark_rate_limit(error, [(is_search, self.search_limited),
                                              (is_index, self.index_limited), (is_ingest, self.ingest_limited)])
                if error.status not in (500, 502, 503, 504) or attempt == attempts - 1:
                    raise
            except (OSError, subprocess.TimeoutExpired):
                if attempt == attempts - 1:
                    raise
            time.sleep(2 ** attempt)
        raise AssertionError("Unreachable")

    def _mark_rate_limit(self, error: APIError, gates: list[tuple[bool, threading.Event]]) -> None:
        if error.status == 429 or (error.status == 403 and "rate limit" in str(error).lower()):
            for enabled, gate in gates:
                if enabled:
                    gate.set()

    def _search_request(self, endpoint: str, body: _JSON) -> tuple[_JSON, float]:
        query = body["prompt"]
        if not query.strip() or len(query.encode("utf-8")) > 7800:
            raise ProtocolError("Query must fit 7800 UTF-8 bytes without rewriting/truncation")
        with self._schedule_lock:
            now = time.monotonic()
            scheduled = max(now, self._next_search)
            self._next_search = scheduled + self.request_interval
        time.sleep(max(0.0, scheduled - now))
        started = time.perf_counter()
        payload = self.request("POST", endpoint, body)
        return payload, (time.perf_counter() - started) * 1000

    def search(self, remote: str, query: str, limit: int) -> tuple[_JSON, float]:
        """Return response and client-observed milliseconds, excluding admission delay."""
        return self._search_request("/embeddings/code/search", {
            "scoping_query": f"repo:{remote}", "prompt": query, "include_embeddings": False,
            "limit": limit, "embedding_model": _MODEL,
        })

    def search_fileset(self, fileset: str, query: str, limit: int) -> tuple[_JSON, float]:
        """Search an explicitly uploaded diff using the same Metis embedding model."""
        return self._search_request("/external/embeddings/code/search", {
            "scoping_query": f"fileset:{fileset}", "prompt": query,
            "limit": limit, "embedding_model": _MODEL,
        })


def _ranked_files(payload: _JSON, context: _JSON, spec: RepoSpec, limit: int) -> tuple[list[str], list[str], int]:
    if payload.get("indexed_commit_sha") != context["indexed_revision"] or payload.get("embedding_model") != _MODEL:
        raise ProtocolError("Search snapshot/model differs from the verified index")
    hits = payload.get("results")
    if not isinstance(hits, list) or len(hits) > limit:
        raise ProtocolError("Invalid result list or returned chunk budget exceeded")
    raw_paths: list[str] = []
    scoped: list[str] = []
    root = spec.benchmark_root.strip("/") if spec.benchmark_root else None
    for hit in hits:
        if not isinstance(hit, dict) or not isinstance(hit.get("location"), dict):
            raise ProtocolError("Search hit has no location object")
        location = hit["location"]
        repository = location.get("repo")
        if (location.get("commit_sha") != context["indexed_revision"] or not isinstance(repository, dict)
                or str(repository.get("nwo", "")).casefold() != context["remote_repo"].casefold()):
            raise ProtocolError("A search hit has a different commit/repository")
        path = location.get("path")
        if not isinstance(path, str) or not path:
            raise ProtocolError("Search hit has no path")
        normalized = PurePosixPath(path)
        if normalized.is_absolute() or ".." in normalized.parts or str(normalized) == ".":
            raise ProtocolError("Search hit is not a repository-relative path")
        path = normalized.as_posix()
        raw_paths.append(path)
        if root is None or path.startswith(root + "/"):
            scoped.append(path)
    return list(dict.fromkeys(scoped))[:_TOP_K], raw_paths, len(scoped)
