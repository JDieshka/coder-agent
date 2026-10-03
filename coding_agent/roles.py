"""Промпты ролей и парсинг ответов моделей в структурированные данные."""
from . import config, context, llm

PLANNER_SYSTEM = (
    llm.JSON_FORMAT_MARKER + "\n"
    "Ты — планировщик разработки (Planner). Получаешь описание проекта. "
    "Составь план на русском: стек (минимальные зависимости), структура файлов, "
    "и декомпозицию на 5-12 МАЛЕНЬКИХ последовательных задач. Каждая задача должна "
    "быть выполнима за один проход кодирования с контекстом <=16K токенов: "
    "один-два файла максимум. Порядок: каркас -> аутентификация -> модель данных -> "
    "личные чаты -> групповые чаты -> тесты.\n"
    "Верни СТРОГО JSON без пояснений:\n"
    '{"plan_md": "полный текст плана в markdown", '
    '"tasks": [{"id": 1, "title": "...", "goal": "что сделать", '
    '"files": ["путь/файл.py"], "acceptance": "как проверить что готово"}]}'
)

REPLAN_SYSTEM = (
    llm.JSON_FORMAT_MARKER + "\n"
    "Ты — планировщик. По текущему плану, списку задач и прогрессу скорректируй очередь: "
    "предложи НОВЫЕ задачи (не переиспользуй id существующих), если чего-то не хватает. "
    'Верни строго JSON: {"new_tasks":[{"id":<новый int>,"title":"...","goal":"...",'
    '"files":["..."],"acceptance":"..."}]}. Если новых задач не нужно — пустой список.'
)

CODER_SYSTEM = (
    llm.JSON_FORMAT_MARKER + "\n"
    "Ты — разработчик (Coder). Пишешь код одной задачей за раз под RTX4060/контекст 16K. "
    "Пиши полный содержимый код без заглушек TODO; интерфейс согласуй с картой проекта. "
    "Для каждого файла верни полное содержимое (overwrite).\n"
    "Верни СТРОГО JSON:\n"
    '{"files":[{"path":"...","content":"..."}],'
    '"notes":"короткое резюме что сделано для progress.md"}\n'
    "ПРАВИЛА ФОРМАТА (нарушение = отказ приёма):\n"
    "1) Никакого текста до и после JSON, никаких markdown-блоков ```.\n"
    "2) Содержимое файлов — ОДНА JSON-строка: каждый перевод строки как \\n, "
    "каждая кавычка как \\\", обратный слэш как \\\\.\n"
    '3) Если файл большой — всё равно один элемент files с полным content.\n'
    '4) Ключи в двойных кавычках, без trailing запятых.'
)

DEBUG_CODER_SYSTEM = CODER_SYSTEM + (
    " Это ИТЕРАЦИЯ ОТЛАДКИ: исправь ошибки из отчёта тестировщика, минимальными правками."
)

TESTER_SYSTEM = (
    llm.JSON_FORMAT_MARKER + "\n"
    "Ты — тестировщик-дебаггер (Tester). Тебе дают задачу, принятые файлы кода и РЕАЛЬНЫЙ "
    "вывод запуска проверок. Оцени: пройдено ли acceptance. Верни СТРОГО JSON:\n"
    '{"passed": true|false, "diagnosis": "если passed=false: конкретные причины и что исправить по файлам, иначе кратко почему принято"}'
)

AUDITOR_SYSTEM = (
    llm.JSON_FORMAT_MARKER + "\n"
    "Ты — аудитор финальной сверки. Сравни ПЛАН и ФАКТЧЕСКОЕ состояние проекта (карта "
    "файлов, статусы задач, последние проверки). Верни СТРОГО JSON:\n"
    '{"summary":"итог на русском", "covered":["пункт плана: реализован (где)"], '
    '"gaps":["пункт плана: НЕ реализовано/частично (что doделать)"]}'
)


def _dump_raw(role: str, content: str):
    """Полный ответ модели — в agent_state/raw_responses/{role}_{ts}.json (для отладки формата)."""
    try:
        import json as _json, os, time as _time
        d = os.path.join(config.STATE_DIR, "raw_responses")
        os.makedirs(d, exist_ok=True)
        fn = os.path.join(d, f"{role}_{int(_time.time()*1000)}.json")
        with open(fn, "w", encoding="utf-8") as f:
            _json.dump({"role": role, "chars": len(content), "text": content},
                       f, ensure_ascii=False)
    except Exception:
        pass


def chat_json(role: str, messages: list[dict], parse, retries: int = 2):
    """Вызывает модель и парсит JSON; при ошибке парсинга просит модель исправить формат."""
    msgs = list(messages)
    last_err = None
    for attempt in range(retries + 1):
        content = llm.chat(role, msgs)
        _dump_raw(role, content)
        try:
            return parse(content)
        except Exception as e:
            last_err = e
            llm._log(f"role={role} PARSE FAILED (attempt {attempt + 1}): {str(e)[:200]}")
            msgs = list(messages) + [
                {"role": "assistant", "content": content[:4000]},
                {"role": "user", "content":
                    f"Твой ответ не является корректным JSON. Ошибка: {str(last_err)[:300]}\n"
                    "Повтори ответ СТРОГО в запрошенном JSON-формате, без пояснений."},
            ]
    raise llm.LLMError(f"Роль '{role}': {retries + 1} попыток, JSON не распознан. Последняя ошибка: {last_err}")


def planner_create(request: str, map_: str) -> tuple[str, list[dict]]:
    msgs = chat_json("planner", [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": f"ЗАПРОС ПРОЕКТА:\n{request}\n\nФАЙЛЫ ПРОЕКТА:\n{map_}"},
    ], _parse_plan)
    plan_md = msgs.get("plan_md", "")
    tasks = msgs.get("tasks", [])
    if not plan_md and tasks:
        plan_md = "\n".join(f"- #{t.get('id')}: {t.get('title', '')}" for t in tasks)
    if not plan_md or not tasks:
        raise llm.LLMError("План пуст или задачи не распознаны")
    return plan_md, _normalize_tasks(tasks)


def _parse_plan(content: str) -> dict:
    data = _coerce_dict(llm.extract_json(content), "tasks")
    if not data.get("tasks"):
        raise llm.LLMError("в JSON нет массива 'tasks'")
    return data


def planner_replan(plan_md: str, tasks_status: str, progress: str, map_: str) -> list[dict]:
    data = chat_json("planner", [
        {"role": "system", "content": REPLAN_SYSTEM},
        {"role": "user", "content": f"ПЛАН:\n{plan_md}\n\nЗАДАЧИ:\n{tasks_status}\n\n"
                                    f"ПРОГРЕСС:\n{progress}\n\nФАЙЛЫ:\n{map_}"},
    ], lambda c: _coerce_dict(llm.extract_json(c), "new_tasks"))
    return _normalize_tasks(data.get("new_tasks", []))


def _coerce_dict(data, key: str) -> dict:
    """Модель может вернуть голый список вместо {"...": [...]} — приводим к словарю."""
    if isinstance(data, list):
        return {key: data}
    if isinstance(data, dict):
        return data
    raise llm.LLMError(f"Ожидался JSON-объект или массив, получено: {type(data).__name__}")


def coder_generate(task: dict, plan_md: str, map_: str, dep_code: str,
                   progress: str, diagnosis: str | None) -> tuple[list[dict], str]:
    system = DEBUG_CODER_SYSTEM if diagnosis else CODER_SYSTEM
    blocks0 = [
        ("ЦЕЛЬ (план)", plan_md[:3000]),
        ("ЗАДАЧА", f"id={task['id']}: {task['title']}\n{task['goal']}\n"
                   f"Файлы: {', '.join(task['files'])}\nAcceptance: {task['acceptance']}"),
        ("КАРТА ПРОЕКТА", map_),
    ]
    blocks = list(blocks0)
    if dep_code:
        blocks.append(("СУЩЕСТВУЮЩИЙ СВЯЗАННЫЙ КОД", dep_code))
    if progress:
        blocks.append(("ПРОГРЕСС", progress))
    if diagnosis:
        blocks.append(("ДИАГНОСТИКА ТЕСТЕРШИКА (исправь!)", diagnosis))
    msgs = context.build_messages(system, blocks)
    data = chat_json("coder", msgs, lambda c: _coerce_dict(llm.extract_json(c), "files"))
    files = data.get("files", [])
    # нормализация: модель могла вернуть {"path": ..., "content": ...} без обёртки "files"
    if not files and task["files"] and "path" in data and "content" in data:
        files = [{"path": data["path"], "content": data["content"]}]
    if not files:
        raise llm.LLMError("Кодер не вернул файлы")
    norm = []
    for f in files:
        if isinstance(f, dict) and f.get("path"):
            norm.append({"path": str(f["path"]), "content": str(f.get("content", ""))})
    if not norm:
        raise llm.LLMError("Кодер вернул файлы без поля 'path'")
    return norm, str(data.get("notes", ""))


def tester_judge(task: dict, written_files: list[str], test_output: str,
                 code_snippets: str) -> tuple[bool, str]:
    blocks = [
        ("ЗАДАЧА", f"{task['title']}\nAcceptance: {task['acceptance']}"),
        ("НАПИСАНЫ ФАЙЛЫ", "\n".join(written_files)),
        ("КОД (фрагменты)", code_snippets),
        ("РЕАЛЬНЫЙ ВЫВОД ПРОВЕРОК (exit code + stdout/stderr)", test_output),
    ]
    data = chat_json("tester", context.build_messages(TESTER_SYSTEM, blocks),
                     lambda c: _coerce_dict(llm.extract_json(c), "verdict"))
    # модель могла вернуть голый true/false или {"passed": ...}
    if isinstance(data, bool):
        return data, ""
    if "passed" not in data and isinstance(data.get("verdict"), dict):
        data = data["verdict"]
    return bool(data.get("passed")), str(data.get("diagnosis", ""))


def auditor_check(plan_md: str, tasks_status: str, map_: str, last_tests: str) -> dict:
    blocks = [("ПЛАН", plan_md), ("СТАТУСЫ ЗАДАЧ", tasks_status),
              ("ФАЙЛЫ", map_), ("ПОСЛЕДНИЕ ПРОВЕРКИ", last_tests)]
    return chat_json("auditor", context.build_messages(AUDITOR_SYSTEM, blocks),
                     llm.extract_json)


def _normalize_tasks(tasks: list[dict]) -> list[dict]:
    out = []
    for i, t in enumerate(tasks):
        out.append({
            "id": int(t.get("id", i + 1)),
            "title": str(t.get("title", f"task-{i+1}")),
            "goal": str(t.get("goal", "")),
            "files": [str(p) for p in t.get("files", [])],
            "acceptance": str(t.get("acceptance", "")),
            "status": "pending",
            "debug_rounds": 0,
        })
    return out
