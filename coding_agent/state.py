"""Состояние: план, очередь задач (tasks.json), прогресс. Восстановление сессии."""
import json
import os

from . import config


def ensure_state():
    os.makedirs(config.STATE_DIR, exist_ok=True)
    for f in (config.LOG_FILE,):
        if not os.path.exists(f):
            open(f, "w").close()


def save_plan(text: str):
    with open(config.PLAN_FILE, "w", encoding="utf-8") as f:
        f.write(text)


def load_plan() -> str:
    try:
        with open(config.PLAN_FILE, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def save_tasks(tasks: list[dict]):
    tmp = config.TASKS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
    os.replace(tmp, config.TASKS_FILE)


def load_tasks() -> list[dict]:
    try:
        with open(config.TASKS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return []


def append_progress(line: str):
    with open(config.PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def load_progress_tail(n_lines: int = 40) -> str:
    try:
        with open(config.PROGRESS_FILE, encoding="utf-8") as f:
            lines = f.readlines()
        return "".join(lines[-n_lines:])
    except OSError:
        return "(прогресс пуст)"
