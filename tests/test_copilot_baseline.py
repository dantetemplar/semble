"""Offline tests for the opt-in remote benchmark; never access GitHub."""

import argparse
import copy
import json
import subprocess
from dataclasses import replace
from unittest.mock import Mock

import pytest

from benchmarks.baselines import copilot
from benchmarks.data import RepoSpec, Target, Task

SHA = "a" * 40
OTHER_SHA = "b" * 40
REMOTE = "demo/repo"


def _spec():
    return RepoSpec("demo", "python", "https://github.com/demo/repo.git", SHA, "src/demo")


def _task(query="find authentication"):
    return Task("demo", "python", query, (Target("src/demo/auth.py"),), (), "semantic")


def _args(**kwargs):
    defaults = dict(revision_policy="exact", index_missing=False, index_timeout=1, poll_interval=1,
                    chunk_limit=30, latency_runs=1, verbose=False, request_interval=0,
                    workers=1, repo=[], language=[], output=None)
    return argparse.Namespace(**{**defaults, **kwargs})


def _status(sha=SHA):
    return {"repository_nwo": REMOTE, "semantic_code_search_ok": True,
            "semantic_commit_sha": sha, "can_index": "ok"}


def _response(paths=None, sha=SHA):
    paths = ["src/demo/auth.py"] if paths is None else paths
    return {"indexed_commit_sha": sha, "embedding_model": copilot._MODEL, "results": [
        {"location": {"path": path, "commit_sha": sha, "repo": {"nwo": REMOTE}}} for path in paths
    ]}


class FakeAPI:
    def __init__(self, sha=SHA, after_sha=None):
        """Initialize a deterministic, network-free repository index."""
        self.sha = sha
        self.after_sha = after_sha or sha
        self.search_calls = []
        self.status_calls = 0
        self.error = None

    def request(self, method, path, body=None):
        """Return repository identity and index status snapshots."""
        if path == "/repos/demo/repo":
            return {"full_name": REMOTE}
        self.status_calls += 1
        return _status(self.sha if self.status_calls == 1 else self.after_sha)

    def search(self, remote, query, limit):
        """Return attributed hits or a configured transport failure."""
        self.search_calls.append((remote, query, limit))
        if self.error:
            raise self.error
        return _response(sha=self.sha), 12.0


def test_transport_keeps_token_out_of_arguments(monkeypatch):
    """Only the subprocess environment receives the token; debug logging is disabled."""
    completed = subprocess.CompletedProcess(
        [], 0, 'HTTP/2.0 200 OK\nContent-Type: application/json\n\n{"ok": true}', '')
    run = Mock(return_value=completed)
    monkeypatch.setattr(copilot.subprocess, "run", run)
    api = copilot.CopilotAPI("test-secret")
    assert api.request("POST", "/embeddings/code/search", {"prompt": "unchanged"}) == {"ok": True}
    command = run.call_args.args[0]
    assert "test-secret" not in " ".join(command)
    assert run.call_args.kwargs["env"]["GH_TOKEN"] == "test-secret"
    assert run.call_args.kwargs["env"]["GH_DEBUG"] == ""
    assert json.loads(run.call_args.kwargs["input"]) == {"prompt": "unchanged"}


def test_transport_preserves_http_errors(monkeypatch):
    """An HTTP failure must not become an empty successful retrieval."""
    completed = subprocess.CompletedProcess(
        [], 1, 'HTTP/2.0 429 Too Many Requests\nRetry-After: 60\n\n{"message":"Rate Limit Exceeded"}', '')
    run = Mock(return_value=completed)
    monkeypatch.setattr(copilot.subprocess, "run", run)
    api = copilot.CopilotAPI("test-secret")
    with pytest.raises(copilot.APIError) as raised:
        api.request("POST", "/embeddings/code/search", {})
    assert raised.value.status == 429
    assert raised.value.headers["retry-after"] == "60"
    with pytest.raises(copilot.APIError, match="paused"):
        api.request("POST", "/embeddings/code/search", {})
    assert run.call_count == 1


def test_transient_reads_retry_but_index_triggers_do_not(monkeypatch):
    """Retry safe reads, not ambiguous index-trigger failures."""
    monkeypatch.setattr(copilot.time, "sleep", Mock())
    api = copilot.CopilotAPI("test-secret")
    once = Mock(side_effect=[ConnectionError("dns"), {"ok": True}])
    monkeypatch.setattr(api, "_once", once)
    assert api.request("GET", "/user") == {"ok": True}
    assert once.call_count == 2
    once = Mock(side_effect=ConnectionError("lost response"))
    monkeypatch.setattr(api, "_once", once)
    with pytest.raises(ConnectionError):
        api.request("POST", "/repos/demo/repo/copilot_internal/embeddings_index", {})
    assert once.call_count == 1


def test_index_quota_does_not_block_already_ready_repositories(monkeypatch):
    """Do not hammer index triggers, but permit reads and searches of ready indexes."""
    api = copilot.CopilotAPI("test-secret")
    once = Mock(side_effect=[copilot.APIError(403, "Rate Limit Exceeded"), _status(), _response()])
    monkeypatch.setattr(api, "_once", once)
    with pytest.raises(copilot.APIError):
        api.request("POST", "/repos/demo/repo/copilot_internal/embeddings_index", {})
    with pytest.raises(copilot.APIError, match="paused"):
        api.request("POST", "/repos/other/repo/copilot_internal/embeddings_index", {})
    assert api.request("GET", "/repos/demo/repo/copilot_internal/embeddings_index") == _status()
    assert api.request("POST", "/embeddings/code/search", {}) == _response()
    assert once.call_count == 3


@pytest.mark.parametrize("query", ["", "я" * 4000])
def test_queries_are_not_silently_truncated(query, monkeypatch):
    """The remote baseline cannot quietly change the benchmark query."""
    api = copilot.CopilotAPI("test-secret", 0)
    request = Mock()
    monkeypatch.setattr(api, "request", request)
    with pytest.raises(copilot.ProtocolError):
        api.search(REMOTE, query, 30)
    request.assert_not_called()


def test_raw_budget_then_root_filter_then_first_file_dedup():
    """Respect benchmark_root boundaries without reordering or gold-aware filtering."""
    paths = ["tests/auth.py", "src/demo-extra/a.py", "src/demo/auth.py", "src/demo/auth.py", "src/demo/b.py"]
    files, raw, in_scope = copilot._ranked_files(
        _response(paths), {"remote_repo": REMOTE, "indexed_revision": SHA}, _spec(), 30)
    assert raw == paths
    assert in_scope == 3
    assert files == ["src/demo/auth.py", "src/demo/b.py"]


def test_empty_results_and_no_benchmark_root():
    """An attributed empty response is a real zero; unrestricted roots keep all paths."""
    context = {"remote_repo": REMOTE, "indexed_revision": SHA}
    assert copilot._ranked_files(_response([]), context, _spec(), 30) == ([], [], 0)
    files, _, _ = copilot._ranked_files(_response(["tests/a.py"]), context, replace(_spec(), benchmark_root=None), 30)
    assert files == ["tests/a.py"]


@pytest.mark.parametrize(
    "kind", ["snapshot", "model", "hit_sha", "hit_repo", "absolute", "traversal", "schema", "budget"]
)
def test_reject_unattributed_or_out_of_budget_results(kind):
    """Never score a response belonging to another snapshot, model or repository."""
    payload = _response()
    if kind == "snapshot":
        payload["indexed_commit_sha"] = OTHER_SHA
    elif kind == "model":
        payload["embedding_model"] = "another-model"
    elif kind == "hit_sha":
        payload["results"][0]["location"]["commit_sha"] = OTHER_SHA
    elif kind == "hit_repo":
        payload["results"][0]["location"]["repo"] = {"nwo": "other/repo"}
    elif kind in ("absolute", "traversal"):
        payload["results"][0]["location"]["path"] = "/tmp/a.py" if kind == "absolute" else "../a.py"
    elif kind == "schema":
        payload["results"] = [None]
    else:
        payload["results"] *= 31
    with pytest.raises(copilot.ProtocolError):
        copilot._ranked_files(payload, {"remote_repo": REMOTE, "indexed_revision": SHA}, _spec(), 30)


def test_all_hits_are_checked_even_beyond_top_ten_files():
    """A bad late hit must not be hidden by the file-result cutoff."""
    payload = _response([f"src/demo/{i}.py" for i in range(20)])
    payload["results"][-1]["location"]["commit_sha"] = OTHER_SHA
    with pytest.raises(copilot.ProtocolError):
        copilot._ranked_files(payload, {"remote_repo": REMOTE, "indexed_revision": SHA}, _spec(), 30)


def test_same_file_level_metrics_as_probe():
    """Use upstream file_rank/ndcg_at_k, with the same treatment of span targets."""
    task = replace(_task(), relevant=(Target("src/demo/auth.py", 100, 200),))
    result = copilot._query(FakeAPI(), _spec(), task, {"remote_repo": REMOTE, "indexed_revision": SHA}, _args())
    assert result["ndcg10"] == 1.0
    assert result["relevant_ranks"] == [1]
    assert result["p50_ms"] == 12.0


def test_exact_policy_rejects_mismatch_and_records_both_revisions():
    """A remote-index result cannot enter exact-mode quality metrics."""
    api = FakeAPI(OTHER_SHA)
    published = []
    copilot._evaluate_repo(api, _spec(), [_task()], _args(), {}, lambda row: published.append(copy.deepcopy(row)))
    row = published[-1]
    assert row["indexed_revision"] == OTHER_SHA
    assert row["pinned_revision"] == SHA
    assert row["queries"][0]["status"] == "error"
    assert not api.search_calls


def test_remote_policy_explicitly_allows_mismatch():
    """Keep mismatched remote snapshots separate and auditable."""
    api = FakeAPI(OTHER_SHA)
    published = []
    copilot._evaluate_repo(api, _spec(), [_task()], _args(revision_policy="remote-index"), {},
                          lambda row: published.append(copy.deepcopy(row)))
    row = published[-1]
    assert row["status"] == "ok"
    assert row["revision_matches_pinned"] is False
    assert row["queries"][0]["indexed_revision"] == OTHER_SHA


def test_after_search_verification_invalidates_changed_index():
    """Do not accept a repo whose index snapshot changed during evaluation."""
    published = []
    copilot._evaluate_repo(FakeAPI(after_sha=OTHER_SHA), _spec(), [_task()], _args(), {},
                          lambda row: published.append(copy.deepcopy(row)))
    row = published[-1]
    assert row["status"] == "partial"
    assert row["queries"][0]["status"] == "error"
    assert copilot._summary([row], 1)["n_successful"] == 0


def test_resume_retries_errors_and_reuses_only_matching_snapshot():
    """Checkpoint reuse needs both the task fingerprint and indexed revision."""
    tasks = [_task(), _task("another query")]
    published = []
    copilot._evaluate_repo(FakeAPI(), _spec(), tasks, _args(), {}, lambda row: published.append(copy.deepcopy(row)))
    previous = published[-1]
    previous["queries"][1]["status"] = "error"
    api = FakeAPI()
    copilot._evaluate_repo(api, _spec(), tasks, _args(), previous, lambda row: None)
    assert len(api.search_calls) == 1
    api = FakeAPI(OTHER_SHA)
    copilot._evaluate_repo(api, _spec(), tasks, _args(revision_policy="remote-index"), previous, lambda row: None)
    assert len(api.search_calls) == 2
    api = FakeAPI()
    copilot._evaluate_repo(api, _spec(), [_task("changed annotation"), tasks[1]], _args(), previous, lambda row: None)
    assert len(api.search_calls) == 2


def test_task_fingerprint_covers_labels_not_only_query():
    """Changing labels must invalidate a checkpoint even when query text is unchanged."""
    task = _task()
    changed = replace(task, secondary=(Target("src/demo/other.py"),))
    assert copilot._task_record(task, 0)["task_id"] != copilot._task_record(changed, 0)["task_id"]


def test_failure_denominators_and_repo_macro_are_explicit():
    """Separate conditional retrieval quality from availability and pending queries."""
    rows = [{"status": "ok", "queries": [{"status": "ok", "ndcg10": 1, "p50_ms": 10}]},
            {"status": "error", "queries": [{"status": "error", "error": "timeout"}]}]
    result = copilot._summary(rows, 3)
    assert result["n_successful"] == result["n_errors"] == result["n_pending"] == 1
    assert result["query_micro_ndcg10"] == result["repo_macro_ndcg10"] == 1
    assert result["end_to_end_ndcg10"] == pytest.approx(1 / 3)
    assert copilot._summary([], 0)["query_micro_ndcg10"] is None


@pytest.mark.parametrize("token", [None, "test-token"])
def test_network_requires_explicit_opt_in(token, monkeypatch):
    """Merely importing or running the baseline must not initiate remote requests."""
    monkeypatch.setattr(copilot.sys, "argv", ["copilot"])
    if token:
        monkeypatch.setenv("GH_TOKEN", token)
    else:
        monkeypatch.delenv("GH_TOKEN", raising=False)
    with pytest.raises(SystemExit) as raised:
        copilot._parse_args()
    assert raised.value.code == 2


def test_token_required_even_after_opt_in(monkeypatch):
    """Authentication must be deliberate and must not be taken from a logged command."""
    monkeypatch.setattr(copilot.sys, "argv", ["copilot", "--include-copilot"])
    monkeypatch.delenv("GH_TOKEN", raising=False)
    with pytest.raises(SystemExit) as raised:
        copilot._parse_args()
    assert raised.value.code == 2


def test_main_does_not_require_local_checkouts(tmp_path, monkeypatch):
    """Unlike local baselines, load all selected annotations, not available_repo_specs."""
    args = _args(output=tmp_path / "results.json")
    monkeypatch.setattr(copilot, "_parse_args", lambda: args)
    monkeypatch.setattr(copilot, "load_repo_specs", lambda: {"demo": _spec()})
    monkeypatch.setattr(copilot, "load_tasks", lambda specs: [_task()])
    monkeypatch.setattr(copilot, "CopilotAPI", lambda *args: FakeAPI())
    monkeypatch.setattr(copilot, "current_sha", lambda: SHA)
    monkeypatch.setenv("GH_TOKEN", "test-token")
    copilot.main()
    result = json.loads(args.output.read_text())
    assert result["status"] == "completed"
    assert result["summary"]["n_successful"] == 1
    assert result["avg_ndcg10"] == 1
    # A matching output resumes without another API call.
    monkeypatch.setattr(copilot, "CopilotAPI", lambda *args: None)
    copilot.main()
    assert len(json.loads(args.output.read_text())["metadata"]["runs"]) == 2


def test_resume_rejects_changed_configuration(tmp_path, monkeypatch):
    """Do not silently combine queries, labels or retrieval budgets from different runs."""
    path = tmp_path / "results.json"
    original = '{"metadata": {"configuration_sha256": "different"}}'
    path.write_text(original)
    monkeypatch.setattr(copilot, "_parse_args", lambda: _args(output=path))
    monkeypatch.setattr(copilot, "load_repo_specs", lambda: {"demo": _spec()})
    monkeypatch.setattr(copilot, "load_tasks", lambda specs: [_task()])
    monkeypatch.setattr(copilot, "current_sha", lambda: SHA)
    with pytest.raises(SystemExit, match="different annotations"):
        copilot.main()
    assert path.read_text() == original
