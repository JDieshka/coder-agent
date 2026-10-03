"""Прогресс-бар для консоли: фазы, задачи, раунды отладки. Работает без зависимостей."""
import os
import shutil
import sys
import time


def _terminal_width(default: int = 80) -> int:
    try:
        return max(40, shutil.get_terminal_size((default, 20)).columns)
    except Exception:
        return default


class ProgressBar:
    """Один переиспользуемый строковый прогресс-бар (\\r), печатается только в TTY.

    Состояние (для resume) можно сохранить/восстановить через snapshot()/restore().
    """

    PHASES = ["plan", "code", "replan", "audit"]

    def __init__(self, log_fn=None):
        self.log = log_fn or (lambda m: None)
        self.total_tasks = 0
        self.done_tasks = 0
        self.current_task_id = 0
        self.current_title = ""
        self.phase = "idle"
        self.rounds = 0
        self.max_rounds = 3
        self._last_render = ""
        self._enabled = sys.stdout.isatty() and os.environ.get("CODER_AGENT_NO_BAR") != "1"
        self._start_ts = time.time()

    # ---------- утилиты ----------
    @staticmethod
    def _fmt_time(sec: float) -> str:
        sec = int(max(0, sec))
        h, rem = divmod(sec, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}ч{m:02d}м"
        if m:
            return f"{m}м{s:02d}с"
        return f"{s}с"

    def elapsed(self) -> str:
        return self._fmt_time(time.time() - self._start_ts)

    def eta(self) -> str:
        proc = self.processed()
        if proc <= 0 or self.total_tasks <= 0:
            return "--"
        per = (time.time() - self._start_ts) / proc
        left = per * max(0, self.total_tasks - proc)
        return self._fmt_time(left)

    # ---------- публичные события ----------
    def set_plan(self, tasks: list[dict]):
        """Синхронизирует счётчики с актуальным tasks.json (идемпотентно для resume/replan)."""
        self.total_tasks = len(tasks)
        self.done_tasks = sum(1 for t in tasks if t.get("status") == "done")
        # failed тоже пересчитываем из задач — иначе при повторных вызовах счётчик накапливался
        self._failed = sum(1 for t in tasks if t.get("status") == "failed")
        # флаги «уже засчитана» синхронизируем со статусами: done/failed -> True,
        # возвращённая в pending задача (реплан) сбрасывает флаг и будет засчитана заново
        for t in tasks:
            setattr(self, f"_counted_{t['id']}", t.get("status") in ("done", "failed"))
        self._render(force=True)

    def start_task(self, task: dict):
        self.current_task_id = task["id"]
        self.current_title = task["title"]
        self.phase = "code"
        self.rounds = 0
        self._render(force=True)

    def finish_task(self, task: dict):
        st = task.get("status")
        # идемпотентность: счётчик ведём по статусу задачи (set_plan пересчитывает из tasks.json);
        # повторный вызов для той же задачи не должен завышать прогресс
        if st == "done" and not getattr(self, f"_counted_{task['id']}", False):
            self.done_tasks += 1
            setattr(self, f"_counted_{task['id']}", True)
        elif st == "failed" and not getattr(self, f"_counted_{task['id']}", False):
            self._failed = getattr(self, "_failed", 0) + 1
            setattr(self, f"_counted_{task['id']}", True)
        self.current_task_id = 0
        self.current_title = ""
        self.phase = "pending"
        self._render(force=True)

    def debug_round(self, rounds: int):
        self.rounds = rounds
        self.phase = "debug"
        self._render(force=True)

    def tick(self, label: str | None = None):
        """Фоновое обновление строки (таймер/ETA) во время долгого ожидания модели."""
        if label is not None and label != self.phase:
            self.phase = label
        self._render(force=True)

    def set_phase(self, phase: str):
        self.phase = phase
        self._render(force=True)

    def processed(self) -> int:
        """Сколько задач уже обработано (done + failed)."""
        return self.done_tasks + self.failed_count()

    def failed_count(self) -> int:
        # точный счётчик храним отдельно; set_plan()/finish_task() держат его синхронно
        return getattr(self, "_failed", 0)

    # ---------- рендер ----------
    def _bar_str(self, ratio: float, width: int) -> str:
        filled = int(round(ratio * width))
        return "█" * filled + "░" * (width - filled)

    def render_line(self) -> str:
        w = _terminal_width() - 2
        label_w = min(28, max(12, w // 4))
        # фиксированный бюджет строки: [100%](7)+bar+пробел+"00/000"(6)+заголовок+фазы+таймеры
        meta = 7 + 1 + 6 + 3 + label_w + 3 + 14 + 3 + 16
        bar_w = max(8, w - meta)
        denom = max(1, self.total_tasks)
        done = min(self.done_tasks, denom)
        proc = min(self.processed(), denom)
        ratio = proc / denom
        pct = int(ratio * 100)
        title = self.current_title[:label_w]
        status_map = {
            "plan": "ПЛАН", "code": "КОД", "debug": f"ОТЛАДКА({self.rounds}/{self.max_rounds})",
            "pending": "…", "replan": "РЕ-ПЛАН", "audit": "СВЕРКА", "idle": "",
            "done": "ГОТОВО", "failed": "ПРОВАЛ",
        }
        status = status_map.get(self.phase, "")
        left = f"[{pct:>3}%] {self._bar_str(ratio, bar_w)}"
        right = f"{proc}/{denom}"
        extra = f" | {title}" if title else ""
        line = f"\r{left} {right:<9}{extra:<{label_w}} | {status:<14} | ⏱ {self.elapsed()} ETA {self.eta()}"
        return line[:w + 1]

    def _render(self, force: bool = False):
        if not self._enabled:
            return
        line = self.render_line()
        if not force and line == self._last_render:
            return
        sys.stdout.write(line)
        sys.stdout.flush()
        self._last_render = line

    def clear(self):
        if not self._enabled:
            return
        sys.stdout.write("\r" + " " * (_terminal_width() - 1) + "\r")
        sys.stdout.flush()
        self._last_render = ""

    def print_final(self, summary: str):
        self.clear()
        print(summary, flush=True)

    # ---------- сериализация (resume) ----------
    def snapshot(self) -> dict:
        return {
            "total_tasks": self.total_tasks,
            "done_tasks": self.done_tasks,
            "failed": getattr(self, "_failed", 0),
            "phase": self.phase,
            "current_task_id": self.current_task_id,
            "current_title": self.current_title,
            "rounds": self.rounds,
            "max_rounds": self.max_rounds,
            "elapsed_sec": int(time.time() - self._start_ts),
        }

    def restore(self, snap: dict):
        self.total_tasks = snap.get("total_tasks", 0)
        self.done_tasks = snap.get("done_tasks", 0)
        self._failed = snap.get("failed", 0)
        self.phase = snap.get("phase", "idle")
        self.current_task_id = snap.get("current_task_id", 0)
        self.current_title = snap.get("current_title", "")
        self.rounds = snap.get("rounds", 0)
        self.max_rounds = snap.get("max_rounds", self.max_rounds)
        # старт сдвигаем так, чтобы elapsed соответствовал снапшоту
        self._start_ts = time.time() - snap.get("elapsed_sec", 0)
