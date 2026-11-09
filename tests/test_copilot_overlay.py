"""Offline coverage of the pinned diff overlay and owned-fileset protocol."""

import base64
import copy
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from benchmarks.baselines import copilot
from benchmarks.baselines import copilot_overlay as overlay
from benchmarks.data import RepoSpec


class IngestAPI:
    def __init__(self):
        """Keep a fake pre-existing workspace alongside the benchmark-owned fileset."""
        self.owned = None
        self.calls = []
        self.capacity = 50
        self.unknown_doc = False

    def request(self, method, path, body=None):
        """Model ingestion phases without HTTP or deletion of existing workspace data."""
        self.calls.append((method, path, copy.deepcopy(body)))
        if method == "GET":
            filesets = [{"name": "existing-user-workspace", "checkpoint": "unrelated"}]
            if self.owned and self.owned.created:
                filesets.append({"name": self.owned.name, "checkpoint": self.owned.checkpoint})
            return {"filesets": filesets, "max_filesets": self.capacity}
        if path == "/external/code/ingest":
            return {} if method == "DELETE" else {"ingest_id": "ingest-1"}
        if path.endswith("/batch"):
            return {"doc_ids": ["unknown"] if self.unknown_doc else list(self.owned.documents)}
        return {}

    def search_fileset(self, name, query, limit):
        """Return one hit explicitly attributed to the uploaded document/checkpoint."""
        doc_id, path = next(iter(self.owned.documents.items()))
        return {"embedding_model": copilot._MODEL, "results": [{
            "location": {"fileset": name, "checkpoint": self.owned.checkpoint, "doc_id": doc_id, "path": path},
            "distance": 0.8, "chunk": {"line_range": {"start": 1, "end": 2}},
        }]}, 10.0


def _fileset(tmp_path):
    prepared = {"checkpoint": "pinned-checkpoint", "geo_filter": "geo", "coded_symbols": [],
                "documents": [{"path": "src/auth.py", "doc_id": "doc-1"}]}
    api = IngestAPI()
    fileset = overlay.ExternalFileset(api, prepared, {"src/auth.py": b"pinned code"})
    fileset.journal = tmp_path / "owned-filesets.jsonl"
    api.owned = fileset
    return api, fileset


def test_fileset_uploads_pinned_bytes_and_deletes_only_its_own_name(tmp_path):
    """Use document IDs bound to our bytes; never evict a pre-existing workspace."""
    api, fileset = _fileset(tmp_path)
    fileset.create()
    fileset.verify()
    hits = fileset.search("authentication", 30)
    assert hits[0]["path"] == "src/auth.py"
    upload = next(body for method, path, body in api.calls if path.endswith("/document"))
    assert base64.b64decode(upload["content"]) == b"pinned code"
    assert upload["file_path"] == "src/auth.py"
    fileset.close()
    deletions = [body for method, _, body in api.calls if method == "DELETE"]
    assert deletions == [{"fileset_name": fileset.name}]
    journal = [json.loads(line) for line in fileset.journal.read_text().splitlines()]
    assert [row["event"] for row in journal] == ["creation-requested-after-absence-check", "created", "deleted"]
    assert all(row["fileset"] == fileset.name for row in journal)


def test_capacity_failure_does_not_cleanup_another_workspace(tmp_path):
    """Unlike the IDE's eviction fallback, benchmarks must not delete unrelated filesets."""
    api, fileset = _fileset(tmp_path)
    api.capacity = 1
    with pytest.raises(copilot.APIError, match="capacity"):
        fileset.create()
    fileset.close()
    assert not any(method == "DELETE" for method, _, _ in api.calls)


def test_unknown_requested_document_fails_instead_of_uploading_empty_content(tmp_path):
    """A server-side reconciliation mismatch must not produce a partial accepted index."""
    api, fileset = _fileset(tmp_path)
    api.unknown_doc = True
    with pytest.raises(copilot.ProtocolError, match="unknown document"):
        fileset.create()
    fileset.close()
    assert not any(path.endswith("/finalize") for _, path, _ in api.calls)
    assert [body for method, _, body in api.calls if method == "DELETE"] == [{"fileset_name": fileset.name}]


@pytest.mark.parametrize("field,value", [("checkpoint", "other"), ("fileset", "other"),
                                         ("doc_id", "unknown"), ("path", "src/other.py")])
def test_external_hits_must_match_owned_pinned_checkpoint(tmp_path, field, value):
    """The fileset namespace alone is not enough to attribute an external result."""
    api, fileset = _fileset(tmp_path)
    response, elapsed = api.search_fileset(fileset.name, "query", 30)
    response["results"][0]["location"][field] = value
    api.search_fileset = Mock(return_value=(response, elapsed))
    with pytest.raises(copilot.ProtocolError):
        fileset.search("query", 30)


def test_existing_fileset_collision_is_never_adopted_or_deleted(tmp_path):
    """A UUID collision must fail before any mutation request."""
    api, fileset = _fileset(tmp_path)
    api.request = Mock(return_value={"filesets": [{"name": fileset.name}], "max_filesets": 50})
    with pytest.raises(copilot.ProtocolError, match="adopt"):
        fileset.create()
    fileset.close()
    assert api.request.call_count == 1


def test_lost_creation_response_remains_owned_and_cleanup_is_journaled(tmp_path):
    """An ambiguous POST must not leave an untracked remote fileset."""
    api, fileset = _fileset(tmp_path)
    original = api.request

    def request(method, path, body=None):
        if method == "POST" and path == "/external/code/ingest":
            raise ConnectionError("response lost after sending request")
        return original(method, path, body)

    api.request = request
    with pytest.raises(ConnectionError):
        fileset.create()
    fileset.close()
    assert [body for method, _, body in api.calls if method == "DELETE"] == [{"fileset_name": fileset.name}]
    assert "creation-requested-after-absence-check" in fileset.journal.read_text()


def test_native_helper_does_not_receive_authentication(monkeypatch):
    """The optional hashing WASM utility has no reason to see GitHub tokens."""
    monkeypatch.setenv("GH_TOKEN", "do-not-pass")
    monkeypatch.setenv("GITHUB_TOKEN", "also-secret")
    process = subprocess.CompletedProcess([], 0, '{"documents": []}', '')
    run = Mock(return_value=process)
    monkeypatch.setattr(overlay.subprocess, "run", run)
    assert overlay.native_helper({"action": "prepare", "files": []}) == {"documents": []}
    assert "GH_TOKEN" not in run.call_args.kwargs["env"]
    assert "GITHUB_TOKEN" not in run.call_args.kwargs["env"]


def test_merge_replaces_stale_files_and_uses_descending_metis_score():
    """Preserve the source's same-model ranking and relative quality cutoff."""
    remote = [{"path": "changed.py", "distance": 0.99}, {"path": "keep.py", "distance": 1.0},
              {"path": "weak.py", "distance": 0.1}]
    local = [{"path": "changed.py", "distance": 0.8}, {"path": "added.py", "distance": 0.7},
             {"path": "not-in-diff.py", "distance": 2.0}]
    ranked = overlay.merge_chunks(remote, local, {"changed.py", "added.py"}, 30)
    assert [hit["path"] for hit in ranked] == ["keep.py", "changed.py", "added.py"]
    assert [hit["distance"] for hit in ranked] == [1.0, 0.8, 0.7]


@pytest.mark.parametrize("score", [None, float("nan"), float("inf"), "0.8"])
def test_noncomparable_scores_are_not_silently_fused(score):
    """Never combine rankings from missing or invalid scores."""
    with pytest.raises(copilot.ProtocolError):
        overlay.merge_chunks([{"path": "a.py", "distance": score}], [], set(), 30)


@pytest.fixture
def git_snapshots(tmp_path):
    """Create two real snapshots covering unchanged, modified, added, deleted and renamed files."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True,
                              capture_output=True, text=True).stdout.strip()

    git("init")
    root = repo / "src" / "demo"
    root.mkdir(parents=True)
    for i in range(10):
        (root / f"keep{i}.py").write_text("unchanged = True\n")
    (root / "auth.py").write_text("remote_version = True\n")
    (root / "deleted.py").write_text("not_in_pinned = True\n")
    (root / "old_name.py").write_text("renamed = True\n")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "remote")
    remote = git("rev-parse", "HEAD")
    (root / "auth.py").write_text("pinned_version = True\n")
    (root / "added.py").write_text("added_in_pinned = True\n")
    (root / "deleted.py").unlink()
    (root / "old_name.py").rename(root / "new_name.py")
    git("add", "-A")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "pinned")
    pinned = git("rev-parse", "HEAD")
    return repo, remote, pinned


class FakeFileset:
    def __init__(self, api, prepared, files):
        """Retain bytes for assertions without making HTTP calls."""
        self.name, self.checkpoint, self.uploaded = "owned-fake", prepared["checkpoint"], len(files)
        self.files = files

    def create(self):
        """Pretend the isolated pinned checkpoint was uploaded."""

    def verify(self):
        """Pretend the checkpoint is unchanged."""

    def close(self):
        """No remote resources are created by this fake."""


def test_git_diff_uses_pinned_bytes_and_covers_deletes_and_renames(git_snapshots, tmp_path, monkeypatch):
    """The overlay's effective corpus must come from the requested Git tree, not the index tip."""
    repo, remote, pinned = git_snapshots
    monkeypatch.setattr(overlay, "_git_cache", lambda *args: repo)
    monkeypatch.setattr(overlay, "ExternalFileset", FakeFileset)

    def prepare(name, files):
        return {"checkpoint": "pinned-files", "cache_version": 1, "skipped": [],
                "documents": [{"path": path, "doc_id": path} for path in sorted(files)]}

    monkeypatch.setattr(overlay, "prepare_documents", prepare)
    spec = RepoSpec("demo", "python", "https://github.com/demo/repo.git", pinned, "src/demo")
    result = overlay.DiffOverlay.prepare(None, spec, {"indexed_revision": remote}, tmp_path / "cache")
    assert result.changed == {"src/demo/auth.py", "src/demo/added.py", "src/demo/deleted.py",
                              "src/demo/old_name.py", "src/demo/new_name.py"}
    assert result.fileset.files == {"src/demo/auth.py": b"pinned_version = True\n",
                                    "src/demo/added.py": b"added_in_pinned = True\n",
                                    "src/demo/new_name.py": b"renamed = True\n"}
    assert len(result.unchanged) == 10
    assert result.metadata["changed_files"] == 5


def test_git_diff_guard_does_not_upload_an_entire_replaced_workspace(tmp_path, monkeypatch):
    """Excessive diffs are explicit errors, not an undocumented remote-only fallback."""
    monkeypatch.setattr(overlay, "_git_cache", lambda *args: tmp_path)
    trees = [{"old.py": ("100644", "a")}, {"new.py": ("100644", "b")}]
    monkeypatch.setattr(overlay, "_tree", lambda *args: trees.pop(0))
    native = Mock()
    monkeypatch.setattr(overlay, "prepare_documents", native)
    spec = RepoSpec("demo", "python", "https://github.com/demo/repo.git", "a" * 40)
    with pytest.raises(copilot.ProtocolError, match="guards"):
        overlay.DiffOverlay.prepare(None, spec, {"indexed_revision": "b" * 40}, tmp_path)
    native.assert_not_called()


def test_overlay_remote_hit_requires_an_identical_pinned_git_blob():
    """A trusted response SHA is not a substitute for unchanged-file content verification."""
    remote, pinned = "a" * 40, "b" * 40
    api = SimpleNamespace(search=Mock(return_value=({"indexed_commit_sha": remote,
        "embedding_model": copilot._MODEL, "results": [{"distance": 0.9, "location": {
            "path": "src/demo/unknown.py", "commit_sha": remote, "repo": {"nwo": "demo/repo"}}}]}, 1.0)))
    spec = RepoSpec("demo", "python", "https://github.com/demo/repo.git", pinned, "src/demo")
    result = overlay.DiffOverlay(api, spec, {"remote_repo": "demo/repo", "indexed_revision": remote})
    with pytest.raises(copilot.ProtocolError, match="identical tracked blob"):
        result.search("query", 30)


def test_overlay_uploads_need_separate_cli_consent(monkeypatch):
    """An existing Copilot token alone must not authorize upload of local source contents."""
    monkeypatch.setattr(copilot.sys, "argv", ["copilot", "--include-copilot", "--revision-policy", "overlay"])
    monkeypatch.setenv("GH_TOKEN", "test-token")
    with pytest.raises(SystemExit) as raised:
        copilot._parse_args()
    assert raised.value.code == 2
