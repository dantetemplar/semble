"""Diff overlay using VS Code's external-ingest protocol, not a local search proxy.

Only public benchmark source files are uploaded. Dedicated UUID filesets are owned
by this run; existing workspace filesets are never replaced or cleaned up.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from benchmarks.baselines.copilot_api import _MODEL, APIError, CopilotAPI, ProtocolError, _ranked_files
from benchmarks.data import RepoSpec

_HELPER = Path(__file__).parents[1] / "copilot_ingest" / "prepare.cjs"
_MAX_BYTES = 64 * 1024 * 1024
_JOURNAL_LOCK = threading.Lock()
_JSON = dict[str, Any]


def native_helper(request: _JSON) -> _JSON:
    """Compute official ingest hashes, filters and coded symbols without credentials."""
    env = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT", "TMPDIR", "LANG", "LC_ALL", "NO_COLOR")
           if key in os.environ}
    process = subprocess.run(["node", str(_HELPER)], input=json.dumps(request),
                             capture_output=True, text=True, timeout=120, check=False, env=env)
    if process.returncode:
        raise ProtocolError(f"Ingest helper failed; run npm ci in benchmarks/copilot_ingest: {process.stderr[-500:]}")
    return json.loads(process.stdout)


def prepare_documents(repo: str, files: dict[str, bytes]) -> _JSON:
    """Apply the official ingest filter and hash the accepted pinned file contents."""
    return native_helper({"action": "prepare", "repo": repo,
                          "files": [{"path": path, "content": base64.b64encode(data).decode()}
                                    for path, data in sorted(files.items())]})


class ExternalFileset:
    """A single dedicated external-ingest fileset with explicit ownership checks."""

    def __init__(self, api: CopilotAPI, prepared: _JSON, files: dict[str, bytes]) -> None:
        """Keep document IDs bound to pinned contents, not arbitrary server paths."""
        self.api = api
        self.prepared = prepared
        self.name = "semble-bench." + uuid.uuid4().hex
        self.checkpoint = prepared["checkpoint"]
        self.documents = {doc["doc_id"]: doc["path"] for doc in prepared["documents"]}
        self.files = files
        self.created = False
        self.creation_attempted = False
        self.uploaded = 0
        self.journal = Path.home() / ".cache" / "semble-copilot" / "owned-filesets.jsonl"

    def _record(self, event: str) -> None:
        with _JOURNAL_LOCK:
            self.journal.parent.mkdir(parents=True, exist_ok=True)
            with self.journal.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"event": event, "fileset": self.name,
                                         "checkpoint": self.checkpoint}) + "\n")

    def _symbols(self, ingest_id: str, initial: _JSON | None) -> None:
        current = initial
        for _ in range(64):
            if current is None:
                return
            coded = native_helper({"action": "symbols", "doc_ids": list(self.documents), "range": current})
            response = self.api.request("POST", "/external/code/ingest/coded_symbols", {
                "ingest_id": ingest_id, "coded_symbol_range": current, "coded_symbols": coded["coded_symbols"],
            })
            current = response.get("next_coded_symbol_range")
        raise ProtocolError("External ingest reconciliation exceeded 64 rounds")

    def _upload_document(self, ingest_id: str, doc_id: str) -> None:
        path = self.documents.get(doc_id)
        if path is None:
            raise ProtocolError("Server requested a document outside the owned pinned snapshot")
        self.api.request("POST", "/external/code/ingest/document", {
            "ingest_id": ingest_id, "doc_id": doc_id, "file_path": path,
            "content": base64.b64encode(self.files[path]).decode(),
        })

    def _upload_missing(self, ingest_id: str) -> None:
        page_token = None
        seen: set[str] = set()
        for _ in range(1024):
            body = {"ingest_id": ingest_id}
            if page_token is not None:
                body["page_token"] = page_token
            batch = self.api.request("POST", "/external/code/ingest/batch", body)
            missing = set(batch.get("doc_ids") or []) - seen
            if not missing.issubset(self.documents):
                raise ProtocolError("External ingest requested unknown document IDs")
            # Keep uploads bounded; errors propagate rather than becoming a partial index.
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda doc: self._upload_document(ingest_id, doc), sorted(missing)))
            self.uploaded += len(missing)
            seen.update(missing)
            page_token = batch.get("next_page_token")
            if page_token is None:
                return
        raise ProtocolError("External ingest pagination exceeded 1024 pages")

    def _finalize(self, ingest_id: str) -> None:
        for attempt in range(3):
            self._upload_missing(ingest_id)
            try:
                self.api.request("POST", "/external/code/ingest/finalize", {"ingest_id": ingest_id})
                return
            except APIError as error:
                if error.status != 412 or attempt == 2:
                    raise
                time.sleep(1)

    def create(self) -> None:
        """Create/reconcile/upload/finalize; never evict another workspace's fileset."""
        existing = self.api.request("GET", "/external/code/ingest")
        names = {entry["name"] for entry in existing.get("filesets", [])}
        if self.name in names:
            raise ProtocolError("Refusing to adopt an existing fileset")
        if len(names) >= existing.get("max_filesets", 50):
            raise APIError(429, "Fileset capacity exhausted; existing workspaces will not be deleted")
        self._record("creation-requested-after-absence-check")
        self.creation_attempted = True
        for attempt in range(3):
            try:
                response = self.api.request("POST", "/external/code/ingest", {
                    "fileset_name": self.name, "new_checkpoint": self.checkpoint,
                    "geo_filter": self.prepared["geo_filter"], "coded_symbols": self.prepared["coded_symbols"],
                })
                self.created = True
                self._record("created")
                ingest_id = response["ingest_id"]
                symbol_range = response.get("coded_symbol_range")
                if not ingest_id and symbol_range == {"start": 0, "end": 0}:
                    return
                if not isinstance(ingest_id, str) or not ingest_id:
                    raise ProtocolError("External ingest returned no ingest ID")
                self._symbols(ingest_id, symbol_range)
                self._finalize(ingest_id)
                return
            except APIError as error:
                if error.status != 409 or attempt == 2:
                    raise

    def search(self, query: str, limit: int) -> list[_JSON]:
        """Search the owned checkpoint with Metis and validate every returned document."""
        try:
            response, _ = self.api.search_fileset(self.name, query, limit)
        except APIError as error:
            if error.status != 404:
                raise
            time.sleep(2)  # Same first-search eventual-consistency workaround as VS Code.
            response, _ = self.api.search_fileset(self.name, query, limit)
        if response.get("embedding_model") != _MODEL or not isinstance(response.get("results"), list):
            raise ProtocolError("External search returned the wrong model/schema")
        if len(response["results"]) > limit:
            raise ProtocolError("External search exceeded the chunk budget")
        output = []
        for hit in response["results"]:
            location = hit.get("location", {})
            path = self.documents.get(location.get("doc_id"))
            if (location.get("fileset") != self.name or location.get("checkpoint") != self.checkpoint
                    or path is None or path != location.get("path")):
                raise ProtocolError("External hit is not from the owned pinned checkpoint")
            output.append({"path": path, "distance": hit.get("distance"), "origin": "external-diff",
                           "line_range": hit.get("chunk", {}).get("line_range")})
        return output

    def verify(self) -> None:
        """Confirm that an empty/nonempty response is still bound to our checkpoint."""
        listing = self.api.request("GET", "/external/code/ingest")
        matches = [item for item in listing.get("filesets", []) if item["name"] == self.name]
        if len(matches) != 1 or matches[0].get("checkpoint") != self.checkpoint:
            raise ProtocolError("Owned external checkpoint changed or disappeared")

    def close(self) -> None:
        """Delete only our confirmed-absent UUID, including an ambiguous lost create response."""
        if not self.creation_attempted:
            return
        try:
            self.api.request("DELETE", "/external/code/ingest", {"fileset_name": self.name})
        except APIError as error:
            if error.status != 404:
                self._record("cleanup-failed")
                raise
        self.created = False
        self.creation_attempted = False
        self._record("deleted")


def _git(repo: Path, *arguments: str, data: bytes | None = None, lazy: bool = True) -> bytes:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    if not lazy:
        env["GIT_NO_LAZY_FETCH"] = "1"
    process = subprocess.run(["git", "-C", str(repo), *arguments], input=data,
                             capture_output=True, timeout=300, check=False, env=env)
    if process.returncode:
        raise ProtocolError(f"Git snapshot operation failed: {process.stderr.decode(errors='replace')[-700:]}")
    return process.stdout


def _git_cache(spec: RepoSpec, indexed: str, directory: Path) -> Path:
    if not re_safe_name(spec.name):
        raise ProtocolError("Unsafe benchmark cache name")
    repo = directory / (spec.name + ".git")
    repo.parent.mkdir(parents=True, exist_ok=True)
    if not repo.exists():
        repo.mkdir()
        _git(repo, "init", "--bare")
        _git(repo, "remote", "add", "origin", spec.url)
        _git(repo, "config", "remote.origin.promisor", "true")
        _git(repo, "config", "remote.origin.partialclonefilter", "blob:none")
    if _git(repo, "remote", "get-url", "origin").decode().strip() != spec.url:
        raise ProtocolError("Existing Git cache belongs to a different upstream")
    for sha in dict.fromkeys((indexed, spec.revision)):
        try:
            _git(repo, "cat-file", "-e", sha + "^{commit}", lazy=False)
        except ProtocolError:
            _git(repo, "fetch", "--no-tags", "--filter=blob:none", "--depth=1", "origin", sha)
        actual = _git(repo, "rev-parse", sha + "^{commit}").decode().strip()
        if actual != sha:
            raise ProtocolError("Git cache did not resolve the requested commit")
    return repo


def re_safe_name(value: str) -> bool:
    """Accept simple benchmark IDs, not arbitrary paths in the shared cache."""
    return bool(value) and value not in (".", "..") and all(c.isalnum() or c in "._-" for c in value)


def _tree(repo: Path, sha: str, root: str | None) -> dict[str, tuple[str, str]]:
    arguments = ["ls-tree", "-rz", "--full-tree", sha]
    if root:
        arguments.extend(["--", root])
    result = {}
    for entry in _git(repo, *arguments).split(b"\0"):
        if not entry:
            continue
        header, path = entry.split(b"\t", 1)
        mode, kind, oid = header.decode().split()
        if kind == "blob":
            result[path.decode("utf-8")] = (mode, oid)
    return result


def _read_blobs(repo: Path, wanted: dict[str, str]) -> dict[str, bytes]:
    if not wanted:
        return {}
    oids = sorted(set(wanted.values()))
    request = ("\n".join(oids) + "\n").encode()
    checks = _git(repo, "cat-file", "--batch-check", data=request, lazy=False).decode().splitlines()
    missing = [line.split()[0] for line in checks if line.endswith(" missing")]
    # Fetch missing public blobs in a single pack, not one lazy HTTP fetch/file.
    if missing:
        _git(repo, "fetch", "--no-tags", "--no-write-fetch-head", "--filter=blob:none", "origin", *missing)
        checks = _git(repo, "cat-file", "--batch-check", data=request, lazy=False).decode().splitlines()
    if any(line.endswith(" missing") for line in checks):
        raise ProtocolError("Pinned Git blob is still missing after the bounded fetch")
    if sum(int(line.split()[2]) for line in checks) > _MAX_BYTES:
        raise ProtocolError("Pinned diff exceeds the 64 MiB upload/memory safety budget")
    stream = io.BytesIO(_git(repo, "cat-file", "--batch", data=request))
    objects = {}
    for expected in oids:
        oid, kind, size = stream.readline().decode().split()
        if oid != expected or kind != "blob":
            raise ProtocolError("Git returned an unexpected object while reading pinned contents")
        contents = stream.read(int(size))
        if len(contents) != int(size) or stream.read(1) != b"\n":
            raise ProtocolError("Truncated Git blob")
        objects[oid] = contents
    return {path: objects[oid] for path, oid in wanted.items()}


def merge_chunks(remote: list[_JSON], local: list[_JSON], changed: set[str], limit: int) -> list[_JSON]:
    """Suppress stale files, then use VS Code's same-model descending score + 0.65 spread."""
    merged = [hit for hit in remote if hit["path"] not in changed]
    merged += [hit for hit in local if hit["path"] in changed]
    for hit in merged:
        value = hit.get("distance")
        if not isinstance(value, int | float) or not math.isfinite(value):
            raise ProtocolError("Metis score is missing/nonfinite; do not merge incomparable rankings")
    ranked = sorted(merged, key=lambda hit: -hit["distance"])[:limit]
    return [hit for hit in ranked if hit["distance"] >= ranked[0]["distance"] * 0.65] if ranked else []


class DiffOverlay:
    """Pinned content via unchanged Git blobs plus an external-ingest diff snapshot."""

    def __init__(self, api: CopilotAPI, spec: RepoSpec, context: _JSON) -> None:
        """Initialize a no-upload overlay; prepare() fills the verified Git diff."""
        self.api, self.spec, self.context = api, spec, context
        self.changed: set[str] = set()
        self.unchanged: dict[str, str] = {}
        self.fileset: ExternalFileset | None = None
        self.metadata: _JSON = {"effective_revision": spec.revision, "changed_files": 0,
                                "strategy": "external-ingest-diff", "ingest_utils_version": "0.3.0",
                                "ranking": "same-model descending score; top chunk budget; 0.65 spread"}

    @classmethod
    def prepare(cls, api: CopilotAPI, spec: RepoSpec, context: _JSON, cache: Path) -> DiffOverlay:
        """Build the diff at the pinned revision; never silently fall back to stale HEAD."""
        overlay = cls(api, spec, context)
        if context["indexed_revision"] == spec.revision:
            overlay.metadata["strategy"] = "already-exact-no-diff"
            return overlay
        repo = _git_cache(spec, context["indexed_revision"], cache)
        remote_all = _tree(repo, context["indexed_revision"], None)
        pinned_all = _tree(repo, spec.revision, None)
        changed_all = {path for path in remote_all.keys() | pinned_all.keys()
                       if remote_all.get(path) != pinned_all.get(path)}
        if len(changed_all) > 2000 or len(changed_all) / max(1, len(pinned_all)) > 0.70:
            raise ProtocolError("Diff exceeds VS Code-like 2000-file/70% guards; no stale-HEAD fallback")
        root = spec.benchmark_root
        pinned = {path: value for path, value in pinned_all.items() if not root or path.startswith(root + "/")}
        overlay.changed = {path for path in changed_all if not root or path.startswith(root + "/")}
        overlay.metadata.update({"workspace_changed_files": len(changed_all),
                                 "workspace_tracked_files": len(pinned_all),
                                 "guard_note": "whole-repository tracked-file counts, not workspace-ignore state"})
        overlay.unchanged = {path: value[1] for path, value in pinned.items()
                             if path not in overlay.changed and value[0] in ("100644", "100755")}
        wanted = {path: pinned[path][1] for path in overlay.changed
                  if path in pinned and pinned[path][0] in ("100644", "100755")}
        files = _read_blobs(repo, wanted)
        prepared = prepare_documents(spec.name, files)
        overlay.metadata.update({"changed_files": len(overlay.changed), "changed_paths": sorted(overlay.changed),
                                 "unchanged_files": len(overlay.unchanged), "accepted_files": prepared["documents"],
                                 "skipped_files": prepared["skipped"], "cache_version": prepared["cache_version"],
                                 "diff_bytes": sum(map(len, files.values()))})
        if prepared["documents"]:
            fileset = ExternalFileset(api, prepared, files)
            overlay.fileset = fileset
            overlay.metadata.update({"fileset": fileset.name, "checkpoint": fileset.checkpoint})
            try:
                fileset.create()
                fileset.verify()
            except BaseException:
                fileset.close()
                raise
            overlay.metadata["uploaded_files"] = fileset.uploaded
        return overlay

    def search(self, query: str, limit: int) -> tuple[list[_JSON], list[str], int]:
        """Get both candidate sets, preserving the pinned-file provenance of each hit."""
        response, _ = self.api.search(self.context["remote_repo"], query, limit)
        _, paths, _ = _ranked_files(response, self.context, self.spec, limit)
        remote = []
        root = self.spec.benchmark_root
        for hit, path in zip(response["results"], paths):
            if root and not path.startswith(root + "/"):
                continue
            if path in self.changed:
                continue
            if self.context["indexed_revision"] != self.spec.revision and path not in self.unchanged:
                raise ProtocolError("Remote hit has no identical tracked blob at the pinned revision")
            remote.append({"path": path, "distance": hit.get("distance"), "origin": "unchanged-remote",
                           "blob_sha": self.unchanged.get(path)})
        local = self.fileset.search(query, limit) if self.fileset else []
        merged = merge_chunks(remote, local, self.changed, limit)
        return merged, paths, len(local)

    def verify(self) -> None:
        """Verify the external checkpoint before accepting repository-level results."""
        if self.fileset:
            self.fileset.verify()

    def close(self) -> None:
        """Remove this run's external fileset, never another workspace's fileset."""
        if self.fileset:
            self.fileset.close()
