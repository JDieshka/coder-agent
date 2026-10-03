"""Оркестратор: автоматическое переключение ролей, итерации отладки, git-коммиты, финальная сверка."""
import json
import os
import subprocess
import sys
import time

from . import config, context, llm, roles, state, tests_runner
from .progress_bar import ProgressBar


def _log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    # перед многострочным логом очищаем строку прогресс-бара, чтобы не было наложений
    if _bar is not None and _bar._enabled:
        _bar.clear()
    print(line, flush=True)
    with open(config.LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


_bar: ProgressBar | None = None  # инициализируется в Orchestrator.__init__


def _wait_with_heartbeat(fn, phase_label: str, interval: float = 5.0):
    """Ждёт завершения fn() (llm.chat блокирует поток), обновляя таймер прогресс-бара,
    чтобы во время долгой генерации строка не выглядела зависшей."""
    box: dict = {}

    def worker():
        try:
            box["result"] = fn()
        except BaseException as e:   # noqa: BLE001 — пробрасываем в основной поток
            box["error"] = e

    import threading
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while t.is_alive():
        t.join(interval)
        if _bar is not None and _bar._enabled:
            _bar.tick(phase_label)
    if "error" in box:
        raise box["error"]
    return box.get("result")


def git(*args: str):
    """git внутри каталога проекта (workflow/<project>)."""
    try:
        subprocess.run(["git", *args], check=False, capture_output=True, timeout=30,
                       cwd=config.PROJECT_DIR or ".")
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


def _normalize_task(t: dict) -> dict:
    """Гарантирует обязательные поля задачи (модель может их не вернуть)."""
    t.setdefault("status", "pending")
    t.setdefault("debug_rounds", 0)
    t.setdefault("files", [])
    t.setdefault("title", f"Задача #{t.get('id', '?')}")
    return t


class Orchestrator:
    def __init__(self, request: str, step_mode: bool = False, review_plan: bool = False):
        global _bar
        self.request = request
        self.step_mode = step_mode
        self.review_plan = review_plan
        state.ensure_state()
        state.save_request(request)
        llm.check_models()
        self.plan_md = state.load_plan()
        self.tasks = [_normalize_task(t) for t in state.load_tasks()]
        # crash-resume: если прошлая сессия упала на задаче in_progress — возвращаем её в pending
        resumed = [t["id"] for t in self.tasks if t["status"] == "in_progress"]
        for t in self.tasks:
            if t["status"] == "in_progress":
                t["status"] = "pending"
        if resumed:
            _log(f"Crash-resume: задачи {resumed} были прерваны — статус сброшен в pending")
            self._save()
        # прогресс-бар: состояние восстанавливаем из снапшота (resume между сессиями)
        _bar = self.bar = ProgressBar(log_fn=_log)
        snap = state.load_bar_snapshot()
        if self.tasks:
            # total/done/failed пересчитываются из актуального tasks.json (надёжнее снапшота);
            # restore() возвращает фазу/таймер прошлой сессии для корректного ETA
            if snap:
                self.bar.restore(snap)
            self.bar.set_plan(self.tasks)
        else:
            # план ещё не готов — показываем "0/?" и фазу плана
            self.bar.phase = "plan"
            self.bar._render(force=True)

    def _save_bar(self):
        try:
            state.save_bar_snapshot(self.bar.snapshot())
        except Exception:
            pass

    # -------- фазы --------
    def phase_plan(self):
        if self.plan_md and self.tasks:
            _log("План уже существует — пропускаю Planner (resume).")
            return
        _log("=== Фаза PLANNER (qwen2.5-coder) ===")
        map_ = context.project_map()
        plan_md, tasks = _wait_with_heartbeat(
            lambda: roles.planner_create(self.request, map_), "plan")
        tasks = [_normalize_task(t) for t in tasks]
        state.save_plan(plan_md)
        state.save_tasks(tasks)
        self.plan_md, self.tasks = plan_md, tasks
        _log(f"План сохранён, задач: {len(tasks)}")
        if self.review_plan:
            self._confirm_plan()

    def _confirm_plan(self):
        """Интерактивное подтверждение плана (--review-plan). Пайп/не-TTY — авто-продолжение."""
        print("\n" + "=" * 60)
        print("ПЛАН РЕАЛИЗАЦИИ:")
        print("=" * 60)
        print(self.plan_md)
        print("=" * 60)
        print("Задачи:")
        print(tasks_status_str(self.tasks))
        print("=" * 60)
        if not sys.stdin.isatty():
            _log("--review-plan: stdin не интерактивный — продолжаю автоматически.")
            return
        while True:
            try:
                ans = input("Подтвердить план? [y]/edit <правка>/skip <id,...>/quit: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = "y"
            if ans in ("", "y", "yes", "д", "да"):
                return
            if ans == "quit":
                _log("Пользователь отменил запуск. Состояние сохранено — продолжите той же командой.")
                raise SystemExit(0)
            if ans.startswith("skip"):
                ids = {int(x) for x in ans[4:].replace(",", " ").split() if x.strip().lstrip("-").isdigit()}
                for t in self.tasks:
                    if t["id"] in ids:
                        t["status"] = "skipped"
                self._save()
                _log(f"Пропущено задач: {sorted(ids)}")
                return
            if ans.startswith("edit"):
                # правка текстом: "edit #3 заменить на ..." — применяем к описанию задачи
                parts = ans.split("#", 1)
                if len(parts) == 2:
                    num, rest = parts[1].split(" ", 1)
                    try:
                        tid = int(num)
                    except ValueError:
                        _log("Формат: edit #<id> <новое описание>")
                        continue
                    for t in self.tasks:
                        if t["id"] == tid:
                            t["description"] = rest.strip()
                            _log(f"Задача #{tid}: описание обновлено")
                    self._save()
                continue
            _log("Команды: y (подтвердить), edit #<id> <текст>, skip <id,id>, quit")

    def _next_pending(self) -> dict | None:
        for t in self.tasks:
            if t["status"] == "pending":
                return t
        return None

    def _save(self):
        state.save_tasks(self.tasks)
        self._save_bar()

    def phase_task_loop(self):
        while True:
            task = self._next_pending()
            if not task:
                break
            task["status"] = "in_progress"
            self._save()
            _log(f"=== Задача #{task['id']}: {task['title']} ===")
            self.bar.start_task(task)
            diagnosis = None
            accepted = False
            while not accepted:
                # CODER (qwen3.5:9b; fallback qwen2.5-coder)
                map_ = context.project_map()
                dep = dependency_code_for(task, map_)
                progress = state.load_progress_tail()
                try:
                    files, notes = _wait_with_heartbeat(
                        lambda t=task, m_=map_, d=dep, pr=progress, dg=diagnosis:
                            roles.coder_generate(t, self.plan_md, m_, d, pr, dg),
                        "debug" if diagnosis else "code")
                except llm.LLMError as e:
                    _log(f"Задача #{task['id']}: ошибка кодера ({str(e)[:150]}) — повтор")
                    task["debug_rounds"] += 1
                    self.bar.debug_round(task["debug_rounds"])
                    if task["debug_rounds"] >= config.MAX_DEBUG_ROUNDS:
                        task["status"] = "failed"
                        self._save()
                        self.bar.finish_task(task)
                        _log(f"Задача #{task['id']} провалена (ошибки формата ответа)")
                        break
                    diagnosis = f"Предыдущая попытка не удалась: {str(e)[:400]}. Верни СТРОГО валидный JSON."
                    self._save()
                    continue
                written = []
                for f in files:
                    path, content = f.get("path"), f.get("content")
                    if not path or content is None:
                        continue
                    if ".." in path or os.path.isabs(path):
                        _log(f"подозрительный путь '{path}' — игнор")
                        continue
                    try:
                        context.write_file(path, content)
                    except (PermissionError, ValueError) as e:
                        _log(f"файл '{path}' отклонён: {e}")
                        continue
                    written.append(path)
                if not written:
                    diagnosis = "Кодер не записал ни одного файла. Верни файлы с корректными путями."
                    task["debug_rounds"] += 1
                    self.bar.debug_round(task["debug_rounds"])
                    if task["debug_rounds"] >= config.MAX_DEBUG_ROUNDS:
                        task["status"] = "failed"
                        self._save()
                        self.bar.finish_task(task)
                        _log(f"Задача #{task['id']} провалена (нет файлов)")
                        break
                    continue

                # РЕАЛЬНЫЕ ПРОВЕРКИ
                hard_ok, hard_out = tests_runner.verify_written(written)

                # TESTER (qwen3:8b) оценивает acceptance по реальному выводу
                snippets = "\n\n".join(
                    f"--- {p} ---\n{context.read_file(p)[:1500]}" for p in written
                )
                try:
                    passed, diag = _wait_with_heartbeat(
                        lambda: roles.tester_judge(task, written, hard_out, snippets), "test")
                except llm.LLMError as e:
                    # не роняем процесс: при ошибке формата доверяемся реальным проверкам
                    _log(f"Задача #{task['id']}: ошибка тестера ({str(e)[:150]}) — решающий exit code")
                    passed, diag = True, ""
                accepted = hard_ok and passed
                if accepted:
                    task["status"] = "done"
                    state.append_progress(f"#{task['id']} {task['title']}: {notes or 'готово'}")
                    git("add", "-A"); git("commit", "-m", f"task #{task['id']}: {task['title']}")
                    self._save()
                    _log(f"Задача #{task['id']} ПРИНЯТА")
                    self.bar.finish_task(task)
                else:
                    task["debug_rounds"] += 1
                    diagnosis = (diag or "") + ("\nРЕАЛЬНЫЕ ОШИБКИ:\n" + hard_out if not hard_ok else "")
                    self._save()
                    if task["debug_rounds"] >= config.MAX_DEBUG_ROUNDS:
                        task["status"] = "failed"
                        self._save()
                        _log(f"Задача #{task['id']} ПРОВАЛЕНА после {config.MAX_DEBUG_ROUNDS} раундов")
                        self.bar.finish_task(task)
                        break
                    _log(f"Задача #{task['id']} раунд отладки {task['debug_rounds']}")
                    self.bar.debug_round(task["debug_rounds"])
            # если задача упала — цикл продолжится со следующей pending
            if self.step_mode:
                _log("Режим --step: задача завершена, останавливаюсь. Продолжи той же командой с --step.")
                return

    def phase_replan(self):
        self.bar.set_phase("replan")
        failed = [t for t in self.tasks if t["status"] == "failed"]
        done = [t for t in self.tasks if t["status"] == "done"]
        if not done:
            _log("Ни одна задача не выполнена — реплан не нужен.")
            return
        _log("=== Ре-план: новые мелкие задачи по пробелам ===")
        try:
            new = _wait_with_heartbeat(
                lambda: roles.planner_replan(self.plan_md, tasks_status_str(self.tasks),
                                             state.load_progress_tail(), context.project_map()),
                "replan")
        except llm.LLMError as e:
            _log(f"Ре-план не удался ({str(e)[:150]}) — пропускаю, идем к сверке.")
            new = []
        existing_ids = {t["id"] for t in self.tasks}
        added = [_normalize_task(t) for t in new if t["id"] not in existing_ids]
        for t in failed:
            # второй шанс после реплана; бара не касаемся — _failed остаётся как статистика сессии,
            # а "обработано" для новых pending-задач пересчитает set_plan()
            t["status"] = "pending"; t["debug_rounds"] = 0
        if added:
            self.tasks.extend(added)
            _log(f"Добавлено задач: {len(added)}")
        else:
            _log("Новых задач нет.")
        self._save()
        self.bar.set_plan(self.tasks)

    def phase_audit(self):
        self.bar.set_phase("audit")
        _log("=== Финальная сверка результата с планом (Auditor=qwen3:8b) ===")
        try:
            result = _wait_with_heartbeat(
                lambda: roles.auditor_check(self.plan_md, tasks_status_str(self.tasks),
                                            context.project_map(), state.load_progress_tail(20)),
                "audit")
        except llm.LLMError as e:
            _log(f"Auditor не смог вернуть JSON ({str(e)[:150]}) — механическая сверка по статусам.")
            done = [t for t in self.tasks if t["status"] == "done"]
            failed = [t for t in self.tasks if t["status"] != "done"]
            result = {
                "summary": f"Механическая сверка: принято {len(done)} из {len(self.tasks)} задач.",
                "covered": [f"#{t['id']} {t['title']}" for t in done],
                "gaps": [f"#{t['id']} {t['title']} (статус: {t['status']})" for t in failed],
            }
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
        self.bar.set_phase("plan")
        self.phase_plan()
        if self.tasks:
            self.bar.set_plan(self.tasks)
        try:
            self.phase_task_loop()
            self.phase_replan()
            self.phase_task_loop()   # добиваем новые/возвращённые задачи
            gaps = self.phase_audit()
            if gaps:
                _log("Есть пробелы — один дополнительный цикл реплана.")
                self.phase_replan()
                self.phase_task_loop()
                gaps = self.phase_audit()
        except llm.LLMError as e:
            # Ollama недоступен/OOM/таймаут — сохраняем состояние,resume продолжит сессией позже
            _log(f"Фатальная ошибка LLM: {e}")
            _log("Состояние сохранено — продолжите той же командой (resume).")
            for t in self.tasks:
                if t["status"] == "in_progress":
                    t["status"] = "pending"
            self._save()
            raise
        done = sum(1 for t in self.tasks if t["status"] == "done")
        failed = sum(1 for t in self.tasks if t["status"] == "failed")
        self.bar.done_tasks = done
        self.bar.total_tasks = len(self.tasks)
        self.bar.phase = "audit"
        self.bar.print_final(
            f"[100%] {'█' * 40} {done + failed}/{len(self.tasks)} | ГОТОВО={done} ПРОВАЛ={failed} "
            f"| ⏱ {self.bar.elapsed()} | ЗАВЕРШЕНО. См. {config.AUDIT_FILE}"
        )
        _log(f"ЗАВЕРШЕНО: задач выполнено {done}/{len(self.tasks)}. См. {config.AUDIT_FILE}")
        self._save_bar()
        return {"done": done, "total": len(self.tasks), "gaps": len(gaps)}


def _early_log(msg: str):
    """Лог до выбора проекта (config.LOG_FILE ещё не определён)."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------- CLI-подкоманды: list / status / diff / reset-task ----------
def _list_projects() -> int:
    root = config.WORKFLOW_ROOT
    if not os.path.isdir(root):
        print("Папка workflow/ не найдена — проектов ещё нет.")
        return 0
    rows = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        st = os.path.join(d, "agent_state")
        if not os.path.isdir(st):
            continue
        try:
            tasks = json.load(open(os.path.join(st, "tasks.json"), encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            tasks = []
        done = sum(1 for t in tasks if t.get("status") == "done")
        failed = sum(1 for t in tasks if t.get("status") == "failed")
        req_f = os.path.join(st, "request.txt")
        req = open(req_f, encoding="utf-8").read().strip()[:50] if os.path.isfile(req_f) else ""
        audit = "есть" if os.path.isfile(os.path.join(st, "audit_report.md")) else "-"
        rows.append((name, f"{done}/{len(tasks)}", failed, audit, req))
    if not rows:
        print("Проектов нет.")
        return 0
    w = max(len(r[0]) for r in rows)
    print(f"{'ПРОЕКТ'.ljust(w)}  {'ГОТОВО':7}  ПРОВАЛ  ОТЧЁТ  ЗАПРОС")
    for name, prog, failed, audit, req in rows:
        print(f"{name.ljust(w)}  {prog:7}  {failed:^5}  {audit:^5}  {req}")
    return 0


def _resolve_project(name_or_request: str) -> str | None:
    """Папка проекта по имени в workflow/ либо по совпадению запроса."""
    cand = os.path.join(config.WORKFLOW_ROOT, name_or_request)
    if os.path.isdir(cand):
        return cand
    found = config.find_existing_project(name_or_request)
    if found:
        return found
    # частичное совпадение имени папки
    if os.path.isdir(config.WORKFLOW_ROOT):
        matches = [n for n in os.listdir(config.WORKFLOW_ROOT) if name_or_request in n]
        if len(matches) == 1:
            return os.path.join(config.WORKFLOW_ROOT, matches[0])
        if len(matches) > 1:
            print(f"Неоднозначно ({', '.join(matches)}) — уточни имя.")
            return None
    return None


def _status_project(name_or_request: str) -> int:
    proj = _resolve_project(name_or_request)
    if not proj:
        print(f"Проект не найден: {name_or_request}")
        return 1
    tf = os.path.join(proj, "agent_state", "tasks.json")
    try:
        tasks = json.load(open(tf, encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print(f"{proj}: задач нет (план ещё не создан)")
        return 0
    icons = {"pending": "⏳", "in_progress": "🔨", "done": "✅", "failed": "❌", "skipped": "⏭"}
    print(f"Проект: {proj}")
    for t in tasks:
        icon = icons.get(t.get("status"), "?")
        rounds = t.get("debug_rounds", 0)
        extra = f" (раунды отладки: {rounds})" if rounds else ""
        print(f"  {icon} #{t.get('id')}: {t.get('title')}{extra}")
    done = sum(1 for t in tasks if t.get("status") == "done")
    print(f"Итого: {done}/{len(tasks)} выполнено")
    return 0


def _diff_project(name_or_request: str, task_id: int | None) -> int:
    proj = _resolve_project(name_or_request)
    if not proj:
        print(f"Проект не найден: {name_or_request}")
        return 1
    cmd = ["git", "-C", proj, "log", "--oneline"]
    if task_id is not None:
        cmd += ["--grep", f"task #{task_id}:", "-1"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        print("git-история пуста или git недоступен.")
        return 1
    commits = r.stdout.strip().splitlines()
    if not commits:
        print("Коммитов не найдено.")
        return 0
    if task_id is None:
        print("История коммитов (--diff <проект> <id> — показать изменения задачи):")
        print(r.stdout)
        return 0
    last = commits[-1].split()[0]
    prev_cmd = ["git", "-C", proj, "log", "--format=%H", "--grep", f"task #{task_id}:", "-2"]
    hashes = subprocess.run(prev_cmd, capture_output=True, text=True, timeout=30).stdout.split()
    base = hashes[1] if len(hashes) > 1 else hashes[0] + "^"
    show = subprocess.run(["git", "-C", proj, "show", "--stat", "-p", last if len(hashes) == 1 else base],
                          capture_output=True, text=True, timeout=60)
    print(show.stdout or "(пусто)")
    return 0


def _reset_task(name_or_request: str, task_id: int) -> int:
    proj = _resolve_project(name_or_request)
    if not proj:
        print(f"Проект не найден: {name_or_request}")
        return 1
    tf = os.path.join(proj, "agent_state", "tasks.json")
    try:
        tasks = json.load(open(tf, encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print("tasks.json не найден/битый.")
        return 1
    changed = False
    for t in tasks:
        if t.get("id") == task_id:
            t["status"] = "pending"
            t["debug_rounds"] = 0
            changed = True
    if not changed:
        print(f"Задача #{task_id} не найдена.")
        return 1
    tmp = tf + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
    os.replace(tmp, tf)
    bf = os.path.join(proj, "agent_state", "bar.json")
    if os.path.isfile(bf):
        try:
            snap = json.load(open(bf, encoding="utf-8"))
            snap.pop(f"_counted_{task_id}", None)
            json.dump(snap, open(bf, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        except (OSError, json.JSONDecodeError):
            pass
    print(f"Задача #{task_id} сброшена в pending. Запусти агент с --resume {os.path.basename(proj)}")
    return 0


def main(argv: list[str] | None = None):
    import argparse
    parser = argparse.ArgumentParser(
        prog="python -m coding_agent",
        description='Кодер-агент: план -> код -> тесты -> сверка (ротация моделей через Ollama).',
    )
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="запустить/продолжить проект (по умолчанию)")
    p_run.add_argument("request", help='описание проекта, например "Напиши чат-приложение..."')
    p_run.add_argument("--name", metavar="SLUG", default=None,
                       help="имя папки проекта в workflow/ (по умолчанию — из запроса)")
    p_run.add_argument("--fresh", action="store_true",
                       help="не подхватывать существующий проект; начать с чистой страницы")
    p_run.add_argument("--resume", metavar="NAME", nargs="?", const="", default=None,
                       help="продолжить проект по имени папки в workflow/ (без имени — поиск по запросу)")
    p_run.add_argument("--step", action="store_true",
                       help="выполнить одну задачу и остановиться (повторный запуск продолжит)")
    p_run.add_argument("--review-plan", action="store_true", dest="review_plan",
                       help="показать план после Planner и спросить подтверждение (y/edit/skip/quit)")

    sub.add_parser("list", help="список проектов в workflow/ с прогрессом")

    p_st = sub.add_parser("status", help="статус задач проекта")
    p_st.add_argument("project", help="имя папки проекта или текст запроса")

    p_df = sub.add_parser("diff", help="git-история проекта или изменения конкретной задачи")
    p_df.add_argument("project", help="имя папки проекта или текст запроса")
    p_df.add_argument("task", nargs="?", type=int, default=None, help="id задачи")

    p_rs = sub.add_parser("reset-task", help="вернуть задачу в pending (после провала)")
    p_rs.add_argument("project", help="имя папки проекта или текст запроса")
    p_rs.add_argument("task", type=int, help="id задачи")

    # обратная совместимость: python -m coding_agent "запрос" == run "запрос"
    known = {"run", "list", "status", "diff", "reset-task"}
    args_list = list(argv) if argv is not None else sys.argv[1:]
    if not args_list or args_list[0] not in known:
        args_list = ["run"] + args_list
    args = parser.parse_args(args_list)

    if args.cmd == "list":
        raise SystemExit(_list_projects())
    if args.cmd == "status":
        raise SystemExit(_status_project(args.project))
    if args.cmd == "diff":
        raise SystemExit(_diff_project(args.project, args.task))
    if args.cmd == "reset-task":
        raise SystemExit(_reset_task(args.project, args.task))

    request = args.request
    fresh = args.fresh
    name = config.slugify(args.name) if args.name else None

    if not fresh:
        existing = config.find_existing_project(request)
        if existing:
            config.init_paths(existing)
            _early_log(f"Продолжаю существующий проект: {existing}/")
        elif args.resume is not None and args.resume.strip():
            # явное resume по имени папки
            cand = os.path.join(config.WORKFLOW_ROOT, config.slugify(args.resume))
            if os.path.isdir(cand):
                config.init_paths(cand)
                _early_log(f"Продолжаю проект по имени (--resume): {config.PROJECT_DIR}/")
            else:
                _early_log(f"--resume: папка {cand}/ не найдена — создаю новый проект")
        elif name and os.path.isdir(os.path.join(config.WORKFLOW_ROOT, name)):
            config.init_paths(os.path.join(config.WORKFLOW_ROOT, name))
            _early_log(f"Продолжаю проект по имени: {config.PROJECT_DIR}/")
    if not config.PROJECT_DIR:
        pid = name or config.slugify(request)
        pid = config.make_project_id(pid)
        project_dir = os.path.join(config.WORKFLOW_ROOT, pid)
        config.init_paths(project_dir)
        _early_log(f"Новый проект: {project_dir}/")

    if fresh:
        for f in (config.PLAN_FILE, config.TASKS_FILE, config.PROGRESS_FILE,
                  config.AUDIT_FILE, config.BAR_FILE):
            if os.path.exists(f):
                os.remove(f)
        print("Состояние очищено (--fresh).")

    os.makedirs(config.PROJECT_DIR, exist_ok=True)
    tests_runner.PROJECT_CWD = config.PROJECT_DIR   # все проверки идут внутри проекта
    if not os.path.isdir(os.path.join(config.PROJECT_DIR, ".git")):
        git("init")
        git("add", "-A")
        git("commit", "-m", "init")
    _log(f"Каталог проекта: {os.path.abspath(config.PROJECT_DIR)}")

    orch = Orchestrator(request, step_mode=args.step, review_plan=args.review_plan)
    res = orch.run()
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
