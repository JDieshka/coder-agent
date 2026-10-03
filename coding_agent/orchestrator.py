"""Оркестратор: автоматическое переключение ролей, итерации отладки, git-коммиты, финальная сверка."""
import json
import os
import subprocess
import sys
import time

from . import config, context, llm, roles, state, tests_runner


def _log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(config.LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def git(*args: str):
    try:
        subprocess.run(["git", *args], check=False, capture_output=True, timeout=30)
    except Exception:
        pass  # git не обязателен


def tasks_status_str(tasks: list[dict]) -> str:
    return "\n".join(
        f"- #{t['id']} [{t['status']}] rounds={t['debug_rounds']}: {t['title']}"
        for t in tasks
    )


def dependency_code_for(task: dict, map_: str) -> str:
    """Собираем код файлов из acceptance/связанных модулей (не больше бюджета)."""
    wanted = []
    for line in map_.splitlines():
        path = line.split(" ")[0]
        if path.endswith(".py") and any(
            w in path.lower() for w in ("app", "main", "db", "models", "auth", "chat", task["files"][0].lower())
        ):
            wanted.append(path)
    chunks, used = [], 0
    for p in wanted[:6]:
        src = context.read_file(p)
        cost = context.approx_tokens(src)
        if used + cost > 5200:
            break
        chunks.append(f"--- {p} ---\n{src}")
        used += cost
    return "\n\n".join(chunks)


class Orchestrator:
    def __init__(self, request: str):
        self.request = request
        state.ensure_state()
        llm.check_models()
        self.plan_md = state.load_plan()
        self.tasks = state.load_tasks()

    # -------- фазы --------
    def phase_plan(self):
        if self.plan_md and self.tasks:
            _log("План уже существует — пропускаю Planner (resume).")
            return
        _log("=== Фаза PLANNER (qwen2.5-coder) ===")
        map_ = context.project_map()
        plan_md, tasks = roles.planner_create(self.request, map_)
        state.save_plan(plan_md)
        state.save_tasks(tasks)
        self.plan_md, self.tasks = plan_md, tasks
        _log(f"План сохранён, задач: {len(tasks)}")

    def _next_pending(self) -> dict | None:
        for t in self.tasks:
            if t["status"] == "pending":
                return t
        return None

    def _save(self):
        state.save_tasks(self.tasks)

    def phase_task_loop(self):
        while True:
            task = self._next_pending()
            if not task:
                break
            task["status"] = "in_progress"
            self._save()
            _log(f"=== Задача #{task['id']}: {task['title']} ===")
            diagnosis = None
            accepted = False
            while not accepted:
                # CODER (qwen3.5:9b; fallback qwen2.5-coder)
                map_ = context.project_map()
                dep = dependency_code_for(task, map_)
                progress = state.load_progress_tail()
                files, notes = roles.coder_generate(task, self.plan_md, map_, dep,
                                                    progress, diagnosis)
                written = []
                for f in files:
                    path, content = f.get("path"), f.get("content")
                    if not path or content is None:
                        continue
                    if ".." in path or os.path.isabs(path):
                        _log(f"подозрительный путь '{path}' — игнор")
                        continue
                    context.write_file(path, content)
                    written.append(path)
                if not written:
                    diagnosis = "Кодер не записал ни одного файла. Верни файлы с корректными путями."
                    task["debug_rounds"] += 1
                    if task["debug_rounds"] >= config.MAX_DEBUG_ROUNDS:
                        task["status"] = "failed"
                        self._save()
                        _log(f"Задача #{task['id']} провалена (нет файлов)")
                        break
                    continue

                # РЕАЛЬНЫЕ ПРОВЕРКИ
                hard_ok, hard_out = tests_runner.verify_written(written)

                # TESTER (qwen3:8b) оценивает acceptance по реальному выводу
                snippets = "\n\n".join(
                    f"--- {p} ---\n{context.read_file(p)[:1500]}" for p in written
                )
                passed, diag = roles.tester_judge(task, written, hard_out, snippets)
                accepted = hard_ok and passed
                if accepted:
                    task["status"] = "done"
                    state.append_progress(f"#{task['id']} {task['title']}: {notes or 'готово'}")
                    git("add", "-A"); git("commit", "-m", f"task #{task['id']}: {task['title']}")
                    self._save()
                    _log(f"Задача #{task['id']} ПРИНЯТА")
                else:
                    task["debug_rounds"] += 1
                    diagnosis = (diag or "") + ("\nРЕАЛЬНЫЕ ОШИБКИ:\n" + hard_out if not hard_ok else "")
                    self._save()
                    if task["debug_rounds"] >= config.MAX_DEBUG_ROUNDS:
                        task["status"] = "failed"
                        self._save()
                        _log(f"Задача #{task['id']} ПРОВАЛЕНА после {config.MAX_DEBUG_ROUNDS} раундов")
                        break
                    _log(f"Задача #{task['id']} раунд отладки {task['debug_rounds']}")
            # если задача упала — цикл продолжится со следующей pending

    def phase_replan(self):
        failed = [t for t in self.tasks if t["status"] == "failed"]
        done = [t for t in self.tasks if t["status"] == "done"]
        if not done:
            _log("Ни одна задача не выполнена — реплан не нужен.")
            return
        _log("=== Ре-план: новые мелкие задачи по пробелам ===")
        new = roles.planner_replan(self.plan_md, tasks_status_str(self.tasks),
                                   state.load_progress_tail(), context.project_map())
        existing_ids = {t["id"] for t in self.tasks}
        added = [t for t in new if t["id"] not in existing_ids]
        for t in failed:
            t["status"] = "pending"; t["debug_rounds"] = 0  # второй шанс после реплана
        if added:
            self.tasks.extend(added)
            _log(f"Добавлено задач: {len(added)}")
        else:
            _log("Новых задач нет.")
        self._save()

    def phase_audit(self):
        _log("=== Финальная сверка результата с планом (Auditor=qwen3:8b) ===")
        result = roles.auditor_check(self.plan_md, tasks_status_str(self.tasks),
                                     context.project_map(), state.load_progress_tail(20))
        with open(config.AUDIT_FILE, "w", encoding="utf-8") as f:
            f.write("# Отчёт о сверке с планом\n\n")
            f.write(f"**Итог:** {result.get('summary','')}\n\n## Реализовано\n")
            for c in result.get("covered", []):
                f.write(f"- {c}\n")
            f.write("\n## Пробелы\n")
            gaps = result.get("gaps", [])
            for g in gaps:
                f.write(f"- {g}\n")
        _log(f"Отчёт: {config.AUDIT_FILE}, пробелов: {len(gaps)}")
        return gaps

    # -------- публичный API --------
    def run(self):
        self.phase_plan()
        self.phase_task_loop()
        self.phase_replan()
        self.phase_task_loop()   # добиваем новые/возвращённые задачи
        gaps = self.phase_audit()
        if gaps:
            _log("Есть пробелы — один дополнительный цикл реплана.")
            self.phase_replan()
            self.phase_task_loop()
            gaps = self.phase_audit()
        done = sum(1 for t in self.tasks if t["status"] == "done")
        _log(f"ЗАВЕРШЕНО: задач выполнено {done}/{len(self.tasks)}. См. {config.AUDIT_FILE}")
        return {"done": done, "total": len(self.tasks), "gaps": len(gaps)}


def main():
    if len(sys.argv) < 2:
        print(__doc__ or "")
        print('usage: python -m coding_agent "описание проекта" [--fresh]')
        sys.exit(1)
    request = sys.argv[1]
    fresh = "--fresh" in sys.argv
    if fresh:
        for f in (config.PLAN_FILE, config.TASKS_FILE, config.PROGRESS_FILE, config.AUDIT_FILE):
            if os.path.exists(f):
                os.remove(f)
        print("Состояние очищено (--fresh).")
    orch = Orchestrator(request)
    res = orch.run()
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
