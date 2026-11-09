"""Opt-in GitHub Copilot remote-index baseline on Semble's own annotations.

This uses undocumented endpoints, a GitHub token and outbound HTTPS. It is not a
local/CPU baseline. Exact revision matching is the default; remote-index results
must be reported separately from the pinned-revision leaderboard.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from urllib.parse import urlsplit

from benchmarks.baselines.copilot_api import _MODEL, _TOP_K, APIError, CopilotAPI, ProtocolError, _ranked_files
from benchmarks.data import (
    RepoSpec,
    Task,
    add_filter_args,
    apply_task_filters,
    current_sha,
    grouped_tasks,
    load_repo_specs,
    load_tasks,
    results_path,
)
from benchmarks.metrics import file_rank, ndcg_at_k

if TYPE_CHECKING:
    from benchmarks.baselines.copilot_overlay import DiffOverlay

_JSON = dict[str, Any]


def _remote_name(url: str) -> str:
    parsed = urlsplit(url)
    name = parsed.path.strip("/").removesuffix(".git")
    if parsed.scheme != "https" or parsed.netloc != "github.com" or not re.fullmatch(r"[\w.-]+/[\w.-]+", name):
        raise ProtocolError("Only public GitHub.com repository URLs are supported")
    return name


def _ready(status: _JSON, remote: str) -> bool:
    if str(status.get("repository_nwo", "")).casefold() != remote.casefold():
        raise ProtocolError("Index status belongs to a different repository")
    sha = status.get("semantic_commit_sha")
    return bool(status.get("semantic_code_search_ok") and isinstance(sha, str)
                and re.fullmatch(r"[0-9a-f]{40}", sha))


def _ensure_index(api: CopilotAPI, remote: str, args: argparse.Namespace) -> tuple[_JSON, float, bool]:
    endpoint = f"/repos/{remote}/copilot_internal/embeddings_index"
    status = api.request("GET", endpoint)
    if _ready(status, remote):
        return status, 0.0, False
    if not args.index_missing:
        raise APIError(412, "Index unavailable; triggering requires --index-missing")
    if status.get("can_index") != "ok":
        raise APIError(403, f"Cannot index repository: {status.get('can_index')}")
    started = time.monotonic()
    api.request("POST", endpoint, {"auto": False})
    while time.monotonic() - started < args.index_timeout:
        time.sleep(args.poll_interval)
        status = api.request("GET", endpoint)
        if _ready(status, remote):
            return status, (time.monotonic() - started) * 1000, True
    raise TimeoutError(f"Index unavailable after {args.index_timeout}s")


def _prepare_repo(api: CopilotAPI, spec: RepoSpec, args: argparse.Namespace) -> _JSON:
    upstream = api.request("GET", f"/repos/{_remote_name(spec.url)}")
    if upstream.get("private"):
        raise ProtocolError("Private repositories are outside this public-source benchmark")
    remote = str(upstream["full_name"])
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", remote):
        raise ProtocolError("Invalid canonical repository name")
    status, lag_ms, triggered = _ensure_index(api, remote, args)
    indexed = status["semantic_commit_sha"]
    return {"remote_repo": remote, "indexed_revision": indexed, "revision_matches_pinned": indexed == spec.revision,
            "index_status_before": status, "index_wait_ms": lag_ms, "index_triggered": triggered}


def _task_record(task: Task, position: int) -> _JSON:
    fingerprint = hashlib.sha256(json.dumps(asdict(task), sort_keys=True).encode()).hexdigest()
    return {"task_id": f"{position}:{fingerprint}", "query": task.query, "category": task.category,
            "query_sha256": hashlib.sha256(task.query.encode()).hexdigest(),
            "targets": [asdict(target) for target in task.all_relevant]}


def _query(api: CopilotAPI, spec: RepoSpec, task: Task, context: _JSON, args: argparse.Namespace,
           overlay: DiffOverlay | None = None) -> _JSON:
    if overlay is not None:
        return _overlay_query(task, context, args, overlay)
    times: list[float] = []
    paths: list[str] = []
    raw: list[str] = []
    scoped = 0
    for _ in range(args.latency_runs):
        payload, elapsed = api.search(context["remote_repo"], task.query, args.chunk_limit)
        paths, raw, scoped = _ranked_files(payload, context, spec, args.chunk_limit)
        times.append(elapsed)
    ranks = [rank for target in task.all_relevant if (rank := file_rank(paths, target.path)) is not None]
    return {"status": "ok", "ndcg10": ndcg_at_k(ranks, len(task.all_relevant), _TOP_K),
            "relevant_ranks": ranks, "files": paths, "raw_paths": raw,
            "returned_chunks": len(raw), "in_scope_chunks": scoped,
            "indexed_revision": context["indexed_revision"], "latencies_ms": times,
            "p50_ms": statistics.median(times)}


def _overlay_query(task: Task, context: _JSON, args: argparse.Namespace, overlay: DiffOverlay) -> _JSON:
    times = []
    chunks: list[_JSON] = []
    raw: list[str] = []
    n_local = 0
    for _ in range(args.latency_runs):
        started = time.perf_counter()
        chunks, raw, n_local = overlay.search(task.query, args.chunk_limit)
        times.append((time.perf_counter() - started) * 1000)
    paths = list(dict.fromkeys(hit["path"] for hit in chunks))[:_TOP_K]
    ranks = [rank for target in task.all_relevant if (rank := file_rank(paths, target.path)) is not None]
    fileset = overlay.fileset
    return {"status": "ok", "ndcg10": ndcg_at_k(ranks, len(task.all_relevant), _TOP_K),
            "relevant_ranks": ranks, "files": paths, "raw_paths": raw, "merged_chunks": chunks,
            "returned_chunks": len(raw), "external_chunks": n_local,
            "indexed_revision": context["indexed_revision"], "effective_revision": overlay.spec.revision,
            "overlay_fileset": fileset.name if fileset else None,
            "overlay_checkpoint": fileset.checkpoint if fileset else None,
            "latencies_ms": times, "p50_ms": statistics.median(times)}


def _error(error: Exception) -> _JSON:
    record: _JSON = {"status": "error", "error": f"{type(error).__name__}: {error}"}
    if isinstance(error, APIError):
        record.update({"http_status": error.status, "retry_after": error.headers.get("retry-after")})
    return record


def _verify_after(api: CopilotAPI, row: _JSON) -> None:
    try:
        after = api.request("GET", f"/repos/{row['remote_repo']}/copilot_internal/embeddings_index")
        row["index_status_after"] = after
        if not _ready(after, row["remote_repo"]) or after["semantic_commit_sha"] != row["indexed_revision"]:
            raise ProtocolError("Index changed while evaluating this repository")
    except (APIError, ProtocolError, OSError, subprocess.TimeoutExpired) as error:
        row["verification_error"] = str(error)
        for query in row["queries"]:
            query.update(_error(error))


def _run_queries(api: CopilotAPI, spec: RepoSpec, tasks: list[Task], args: argparse.Namespace,
                 previous: _JSON, row: _JSON, overlay: DiffOverlay | None,
                 publish: Callable[[_JSON], None]) -> None:
    cached = {q["task_id"]: q for q in previous.get("queries", []) if q["status"] == "ok"}
    if (previous.get("indexed_revision") != row["indexed_revision"]
            or previous.get("overlay", {}).get("checkpoint") != row.get("overlay", {}).get("checkpoint")):
        cached = {}
    for position, task in enumerate(tasks):
        record = _task_record(task, position)
        if record["task_id"] in cached:
            record = copy.deepcopy(cached[record["task_id"]])
        else:
            try:
                record.update(_query(api, spec, task, row, args, overlay))
            except (APIError, ProtocolError, OSError, subprocess.TimeoutExpired) as error:
                record.update(_error(error))
        row["queries"].append(record)
        publish(row)
        if args.verbose:
            print(f"  {spec.name}: {task.query!r}: {record.get('ndcg10', record.get('error'))}", file=sys.stderr)


def _evaluate_repo(api: CopilotAPI, spec: RepoSpec, tasks: list[Task], args: argparse.Namespace,
                   previous: _JSON, publish: Callable[[_JSON], None]) -> None:
    row: _JSON = {"repo": spec.name, "language": spec.language, "pinned_revision": spec.revision,
                  "benchmark_root": spec.benchmark_root, "n_tasks": len(tasks), "queries": []}
    overlay = None
    try:
        row.update(_prepare_repo(api, spec, args))
        if args.revision_policy == "exact" and not row["revision_matches_pinned"]:
            raise ProtocolError(f"Revision mismatch: pinned={spec.revision}, remote={row['indexed_revision']}")
        if args.revision_policy == "overlay":
            from benchmarks.baselines.copilot_overlay import DiffOverlay
            started = time.perf_counter()
            overlay = DiffOverlay.prepare(api, spec, row, args.overlay_cache)
            row["overlay"] = {**overlay.metadata, "prepare_ms": (time.perf_counter() - started) * 1000}
    except (APIError, ProtocolError, OSError, subprocess.TimeoutExpired, TimeoutError) as error:
        row["queries"] = [{**_task_record(task, i), **_error(error)} for i, task in enumerate(tasks)]
        row["status"] = "error"
        publish(row)
        return
    try:
        _run_queries(api, spec, tasks, args, previous, row, overlay, publish)
        _verify_after(api, row)
        if overlay is not None:
            _verify_overlay(overlay, row)
    finally:
        if overlay is not None:
            try:
                overlay.close()
                row["overlay"]["cleanup"] = "completed"
            except (APIError, ProtocolError, OSError, subprocess.TimeoutExpired) as error:
                row["overlay"]["cleanup_error"] = str(error)
    row["status"] = "ok" if all(q["status"] == "ok" for q in row["queries"]) else "partial"
    publish(row)


def _verify_overlay(overlay: DiffOverlay, row: _JSON) -> None:
    try:
        overlay.verify()
    except (APIError, ProtocolError, OSError, subprocess.TimeoutExpired) as error:
        row["verification_error"] = str(error)
        for query in row["queries"]:
            query.update(_error(error))


def _summary(rows: list[_JSON], total_queries: int) -> _JSON:
    queries = [query for row in rows for query in row["queries"]]
    successful = [query for query in queries if query["status"] == "ok"]
    completed = [row for row in rows if row.get("status") == "ok"]
    means = [statistics.mean(q["ndcg10"] for q in row["queries"]) for row in completed]
    return {"n_queries_requested": total_queries, "n_queries_attempted": len(queries),
            "n_successful": len(successful), "n_errors": len(queries) - len(successful),
            "n_pending": total_queries - len(queries), "n_repos_complete": len(completed),
            "query_micro_ndcg10": statistics.mean(q["ndcg10"] for q in successful) if successful else None,
            "repo_macro_ndcg10": statistics.mean(means) if means else None,
            "end_to_end_ndcg10": sum(q["ndcg10"] for q in successful) / total_queries if total_queries else None,
            "query_p50_ms": statistics.median(q["p50_ms"] for q in successful) if successful else None}


def _payload(metadata: _JSON, rows: list[_JSON], status: str) -> _JSON:
    summary = _summary(rows, metadata["n_queries"])
    exact = [row for row in rows if row.get("revision_matches_pinned")]
    pinned = [row for row in rows if row.get("revision_matches_pinned") or "overlay" in row]
    return {"tool": "copilot-overlay" if metadata["revision_policy"] == "overlay" else "copilot-remote",
            "status": status, "metadata": metadata,
            "avg_ndcg10": summary["repo_macro_ndcg10"], "summary": summary,
            "exact_revision_subset": _summary(exact, sum(row["n_tasks"] for row in exact)),
            "pinned_corpus_subset": _summary(pinned, sum(row["n_tasks"] for row in pinned)),
            "by_language": {language: _summary([r for r in rows if r["language"] == language], count)
                            for language, count in metadata["queries_by_language"].items()},
            "repos": rows}


def _save(path: Path, payload: _JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_filter_args(parser, verbose=True)
    parser.add_argument("--include-copilot", action="store_true", help="Explicitly allow authenticated remote requests")
    parser.add_argument("--revision-policy", choices=("exact", "remote-index", "overlay"), default="exact")
    parser.add_argument("--upload-diff", action="store_true", help="Consent to uploading public pinned diff files")
    parser.add_argument("--overlay-cache", type=Path,
                        default=Path.home() / ".cache" / "semble-copilot" / "git")
    parser.add_argument("--index-missing", action="store_true", help="Opt in to triggering missing indexes")
    parser.add_argument("--index-timeout", type=float, default=120)
    parser.add_argument("--poll-interval", type=float, default=5)
    parser.add_argument("--chunk-limit", type=int, default=30,
                        help="Returned chunk budget before scope filtering/file dedup")
    parser.add_argument("--latency-runs", type=int, default=1)
    parser.add_argument("--request-interval", type=float, default=0.5,
                        help="Minimum global interval between query starts")
    parser.add_argument("--workers", type=int, default=3, help="Concurrent repositories, 1..4")
    parser.add_argument("--output", type=Path, help="Checkpoint JSON; matching existing output resumes automatically")
    args = parser.parse_args()
    if not args.include_copilot:
        parser.error("Remote baseline is opt-in: pass --include-copilot")
    if args.revision_policy == "overlay" and not args.upload_diff:
        parser.error("Diff uploads require --upload-diff; no stale remote-only fallback is used")
    if not os.environ.get("GH_TOKEN"):
        parser.error("GH_TOKEN is required; it is never written to results")
    if not shutil.which("gh"):
        parser.error("GitHub CLI (gh) is required")
    if not (10 <= args.chunk_limit <= 200 and 1 <= args.workers <= 4 and args.latency_runs >= 1
            and args.request_interval >= 0 and args.index_timeout > 0 and args.poll_interval >= 1):
        parser.error("Invalid chunk/worker/run counts or timeout/interval")
    return args


def _configuration(specs: dict[str, RepoSpec], groups: dict[str, list[Task]], args: argparse.Namespace) -> _JSON:
    inputs = {"specs": [asdict(specs[name]) for name in sorted(groups)],
              "tasks": [asdict(task) for name in sorted(groups) for task in groups[name]],
              "revision_policy": args.revision_policy, "chunk_limit": args.chunk_limit,
              "latency_runs": args.latency_runs, "top_k": _TOP_K, "model": _MODEL,
              "scoring": "file_rank + ndcg_at_k (same file-level protocol as probe; spans ignored)"}
    if args.revision_policy == "overlay":
        helper = Path(__file__).parents[1] / "copilot_ingest" / "prepare.cjs"
        inputs["overlay_protocol"] = {"version": 1, "utilities": "0.3.0", "spread": 0.65,
                                      "helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
                                      "max_diff_files": 2000, "max_diff_fraction": 0.70,
                                      "per_source_chunk_limit": args.chunk_limit}
    fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    languages = {specs[name].language for name in groups}
    return {"configuration_sha256": fingerprint, "benchmark_commit": current_sha(), "model": _MODEL,
            "revision_policy": args.revision_policy, "chunk_limit": args.chunk_limit,
            "file_limit": _TOP_K, "latency_runs": args.latency_runs, "scoring": inputs["scoring"],
            "overlay_protocol": inputs.get("overlay_protocol"),
            "n_repos": len(groups), "n_queries": sum(map(len, groups.values())),
            "queries_by_language": {lang: sum(len(ts) for name, ts in groups.items() if specs[name].language == lang)
                                    for lang in sorted(languages)},
            "note": "Remote service, not CPU-only. benchmark_root is post-filtered after fixed-budget retrieval; "
                    "corpus filters/chunking differ. Latency includes gh/network/retries, excludes admission delay. "
                    "Index wait is not index build time. Conditional metrics exclude explicit errors. "
                    "Overlay mode includes both search calls/admission delay and has an additional candidate source; "
                    "its public-source uploads require explicit consent and its preparation time is separate."}


def main() -> None:
    """Run Copilot on the benchmark annotations without requiring local checkouts."""
    args = _parse_args()
    specs = load_repo_specs()
    tasks = apply_task_filters(load_tasks(specs), repos=args.repo or None, languages=args.language or None)
    if not tasks:
        raise SystemExit("No benchmark tasks matched the requested filters")
    groups = grouped_tasks(tasks)
    metadata = _configuration(specs, groups, args)
    output = args.output or results_path("copilot-" + args.revision_policy)
    existing: _JSON = json.loads(output.read_text()) if output.exists() else {}
    if existing and existing.get("metadata", {}).get("configuration_sha256") != metadata["configuration_sha256"]:
        raise SystemExit("Refusing to resume output with different annotations/specs/scoring configuration")
    metadata["runs"] = [*existing.get("metadata", {}).get("runs", []), {
        "started_utc": datetime.now(timezone.utc).isoformat(), "workers": args.workers,
        "request_interval_s": args.request_interval, "index_missing": args.index_missing,
        "index_timeout_s": args.index_timeout, "benchmark_commit": current_sha(),
    }]
    by_repo = {row["repo"]: row for row in existing.get("repos", [])}
    lock = threading.Lock()

    def publish(row: _JSON) -> None:
        with lock:
            by_repo[row["repo"]] = copy.deepcopy(row)
            _save(output, _payload(metadata, sorted(by_repo.values(), key=lambda r: r["repo"]), "running"))

    api = CopilotAPI(os.environ["GH_TOKEN"], args.request_interval)
    _save(output, _payload(metadata, list(by_repo.values()), "running"))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_evaluate_repo, api, specs[name], repo_tasks, args, by_repo.get(name, {}), publish)
                   for name, repo_tasks in sorted(groups.items()) if by_repo.get(name, {}).get("status") != "ok"]
        for future in futures:
            future.result()
    result = _payload(metadata, sorted(by_repo.values(), key=lambda r: r["repo"]), "completed")
    _save(output, result)
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved {output}", file=sys.stderr)
    if result["summary"]["n_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
