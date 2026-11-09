"""Regression coverage for module execution and remote-overlay error boundaries."""

import json
import runpy
from unittest.mock import Mock

import pytest

from benchmarks import data
from benchmarks.baselines import copilot, copilot_api, copilot_overlay
from benchmarks.data import RepoSpec, Target, Task


def test_shared_protocol_error_identity():
    """Running the driver with -m must not produce a second incompatible error class."""
    assert copilot.ProtocolError is copilot_api.ProtocolError is copilot_overlay.ProtocolError
    assert copilot.ProtocolError.__module__ == "benchmarks.baselines.copilot_api"


def test_module_execution_checkpoints_overlay_preparation_failure(tmp_path, monkeypatch):
    """A real __main__ execution records overlay errors instead of escaping its handler."""
    remote = "a" * 40
    pinned = "b" * 40
    spec = RepoSpec("demo", "python", "https://github.com/demo/repo.git", pinned, "src/demo")
    task = Task("demo", "python", "authentication", (Target("src/demo/auth.py"),), (), "semantic")
    api = Mock()
    api.request.side_effect = [
        {"full_name": "demo/repo", "private": False},
        {"repository_nwo": "demo/repo", "semantic_code_search_ok": True,
         "semantic_commit_sha": remote, "can_index": "ok"},
    ]
    monkeypatch.setattr(copilot_api, "CopilotAPI", lambda *args: api)
    monkeypatch.setattr(data, "load_repo_specs", lambda: {"demo": spec})
    monkeypatch.setattr(data, "load_tasks", lambda specs: [task])
    monkeypatch.setattr(data, "current_sha", lambda: remote)
    monkeypatch.setattr(copilot.shutil, "which", lambda executable: "/fake/gh")
    monkeypatch.setattr(copilot_overlay.DiffOverlay, "prepare",
                        Mock(side_effect=copilot_api.ProtocolError("simulated preparation failure")))
    monkeypatch.setenv("GH_TOKEN", "test-token")
    output = tmp_path / "result.json"
    monkeypatch.setattr(copilot.sys, "argv", ["copilot", "--include-copilot", "--revision-policy", "overlay",
                                              "--upload-diff", "--workers", "1", "--output", str(output)])
    with pytest.warns(RuntimeWarning, match="found in sys.modules"), pytest.raises(SystemExit) as raised:
        runpy.run_module("benchmarks.baselines.copilot", run_name="__main__")
    assert raised.value.code == 1
    result = json.loads(output.read_text())
    assert result["status"] == "completed"
    assert result["summary"]["n_errors"] == 1
    assert result["summary"]["n_successful"] == 0
    assert "simulated preparation failure" in result["repos"][0]["queries"][0]["error"]
    api.search.assert_not_called()


def test_private_upstream_is_rejected_before_search():
    """Explicit diff-upload consent is limited to this public benchmark suite."""
    api = Mock()
    api.request.return_value = {"full_name": "demo/private", "private": True}
    spec = RepoSpec("demo", "python", "https://github.com/demo/private.git", "a" * 40)
    with pytest.raises(copilot.ProtocolError, match="Private repositories"):
        copilot._prepare_repo(api, spec, None)
    assert api.request.call_count == 1
