"""Юнит-тесты прогресс-бара: идемпотентность учёта, skipped, snapshot/resume."""
from coding_agent.progress_bar import ProgressBar


def mkbar():
    bar = ProgressBar(log_fn=lambda m: None)
    bar._enabled = False   # без рендера в не-TTY
    return bar


def tasks(*statuses, ids=None):
    ids = ids or list(range(1, len(statuses) + 1))
    return [{"id": i, "title": f"T{i}", "status": s, "debug_rounds": 0, "files": []}
            for i, s in zip(ids, statuses)]


def test_set_plan_counts():
    bar = mkbar()
    bar.set_plan(tasks("done", "failed", "pending"))
    assert bar.total_tasks == 3 and bar.done_tasks == 1
    assert bar._failed == 1


def test_skipped_excluded_from_total():
    bar = mkbar()
    bar.set_plan(tasks("done", "skipped", "pending"))
    assert bar.total_tasks == 2 and bar.done_tasks == 1


def test_finish_task_idempotent():
    bar = mkbar()
    ts = tasks("in_progress", "pending")
    bar.set_plan(ts)
    ts[0]["status"] = "done"
    bar.finish_task(ts[0])
    bar.finish_task(ts[0])   # повторный вызов не должен удваивать
    assert bar.done_tasks == 1


def test_failed_not_double_counted():
    bar = mkbar()
    ts = tasks("in_progress")
    bar.set_plan(ts)
    ts[0]["status"] = "failed"
    bar.finish_task(ts[0])
    bar.finish_task(ts[0])
    assert bar._failed == 1


def test_snapshot_restore_roundtrip():
    bar = mkbar()
    bar.set_plan(tasks("done", "pending"))
    snap = bar.snapshot()
    bar2 = mkbar()
    bar2.restore(snap)
    bar2.set_plan(tasks("done", "pending"))
    assert bar2.total_tasks == 2 and bar2.done_tasks == 1


def test_replan_resets_counted_flag():
    bar = mkbar()
    ts = tasks("pending")
    bar.set_plan(ts)
    ts[0]["status"] = "failed"
    bar.finish_task(ts[0])
    assert bar._failed == 1
    # реплан вернул задачу в pending — счётчик обнуляется, флаг сброшен
    ts[0]["status"] = "pending"
    bar.set_plan(ts)
    assert getattr(bar, "_counted_1", False) is False
