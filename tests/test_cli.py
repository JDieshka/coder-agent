"""Тесты CLI-подкоманд: list/status/diff/reset-task + обратная совместимость run."""
import json
import os

import pytest

from coding_agent import config
from coding_agent.orchestrator import main


@pytest.fixture()
def wf(tmp_path, monkeypatch):
    """Временный workflow/ с одним проектом и задачами."""
    root = str(tmp_path / "workflow")
    proj = os.path.join(root, "demo-chat")
    st = os.path.join(proj, "agent_state")
    os.makedirs(st)
    open(os.path.join(st, "request.txt"), "w", encoding="utf-8").write("чат\n")
    tasks = [
        {"id": 1, "title": "Каркас", "status": "done", "debug_rounds": 0, "files": ["app.py"]},
        {"id": 2, "title": "Аутентификация", "status": "failed", "debug_rounds": 3, "files": []},
        {"id": 3, "title": "Групповые чаты", "status": "pending", "debug_rounds": 0, "files": []},
    ]
    json.dump(tasks, open(os.path.join(st, "tasks.json"), "w", encoding="utf-8"))
    snap = {"total_tasks": 3, "done_tasks": 1, "_failed": 1, "_counted_2": True}
    json.dump(snap, open(os.path.join(st, "bar.json"), "w", encoding="utf-8"))
    monkeypatch.setattr(config, "WORKFLOW_ROOT", root)
    return root


def run_cli(argv, capsys):
    with pytest.raises(SystemExit) as e:
        main(argv)
    return e.value.code or 0, capsys.readouterr().out


def test_list(wf, capsys):
    code, out = run_cli(["list"], capsys)
    assert code == 0 and "demo-chat" in out and "1/3" in out


def test_status_by_name(wf, capsys):
    code, out = run_cli(["status", "demo-chat"], capsys)
    assert code == 0 and "✅ #1" in out and "❌ #2" in out and "⏳ #3" in out


def test_status_by_request(wf, capsys):
    code, out = run_cli(["status", "чат"], capsys)
    assert code == 0 and "Итого: 1/3" in out


def test_status_not_found(wf, capsys):
    code, out = run_cli(["status", "нет-такого"], capsys)
    assert code == 1


def test_reset_task(wf, capsys):
    code, out = run_cli(["reset-task", "demo-chat", "2"], capsys)
    assert code == 0 and "сброшена" in out
    tasks = json.load(open(os.path.join(wf, "demo-chat", "agent_state", "tasks.json"), encoding="utf-8"))
    assert tasks[1]["status"] == "pending" and tasks[1]["debug_rounds"] == 0
    snap = json.load(open(os.path.join(wf, "demo-chat", "agent_state", "bar.json"), encoding="utf-8"))
    assert "_counted_2" not in snap   # флаг снят — задача будет засчитана заново


def test_reset_task_unknown_id(wf, capsys):
    code, out = run_cli(["reset-task", "demo-chat", "99"], capsys)
    assert code == 1


def test_diff_no_git_history(wf, capsys):
    # в тестовом проекте нет .git — команда не падает, а сообщает
    code, out = run_cli(["diff", "demo-chat"], capsys)
    assert code in (0, 1) and out.strip() != ""
