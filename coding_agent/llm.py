"""Клиент Ollama: вызовы с ротацией моделей (keep_alive=0) и парсингом JSON."""
import json
import re
import time

import requests

from . import config

# явный маркер JSON-режима: добавляется к system-промптам ролей, ожидающих JSON.
# chat() включает format=json по наличию маркера (плюс обратная совместимость со старыми промптами).
JSON_FORMAT_MARKER = "[FORMAT:JSON]"


class LLMError(RuntimeError):
    pass


def _log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(config.LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def list_models() -> set[str]:
    try:
        r = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=10)
        r.raise_for_status()
        return {m["name"] for m in r.json().get("models", [])}
    except Exception as e:
        raise LLMError(f"Ollama недоступен по адресу {config.OLLAMA_URL}: {e}")


def check_models():
    available = list_models()
    missing = []
    for role, rc in config.ROLES.items():
        if rc.model not in available and rc.model not in missing:
            missing.append((role, rc.model))
    if missing:
        names = ", ".join(f"{r}:{m}" for r, m in missing)
        raise LLMError(
            f"Модели не найдены в Ollama: {names}. Выполните `ollama pull <имя>`."
        )


def unload_model(model: str):
    """Явная выгрузка модели из VRAM (keep_alive=0 с пустым запросом) — перед fallback."""
    try:
        requests.post(f"{config.OLLAMA_URL}/api/generate",
                      json={"model": model, "prompt": "", "keep_alive": "0"}, timeout=15)
    except Exception:
        pass


def chat(role: str, messages: list[dict]) -> str:
    """Один «автоматическая сессия»: загрузить модель -> запрос -> выгрузить."""
    rc = config.ROLES[role]
    models_to_try = [rc.model] + ([rc.fallback_model] if rc.fallback_model else [])
    last_err = None
    for attempt in range(2):  # сетевые сбои Ollama ретраим один раз
        for i, model in enumerate(models_to_try):
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "keep_alive": config.KEEP_ALIVE,   # немедленная выгрузка -> следующая роль без OOM
                "options": {
                    "temperature": rc.temperature,
                    "num_ctx": config.CTX_WINDOW,
                    "num_predict": rc.max_tokens,
                },
            }
            if rc.think is False and model.startswith("qwen3"):
                payload["think"] = False  # qwen3: отключаем reasoning-токены для экономии контекста
            wants_json = _wants_json(messages)
            if wants_json:
                # JSON-режим Ollama (grammar-constrained decoding): ответ всегда валидный JSON,
                # переводы строк внутри content экранируются моделью принудительно
                payload["format"] = "json"
            _log(f"role={role} model={model} msg_chars={sum(len(m['content']) for m in messages)}")
            t0 = time.time()
            try:
                r = requests.post(f"{config.OLLAMA_URL}/api/chat", json=payload,
                                  timeout=config.REQ_TIMEOUT)
                r.raise_for_status()
                data = r.json()
                content = data.get("message", {}).get("content", "")
                if not content.strip():
                    raise LLMError("пустой ответ модели")
                _log(f"role={role} done in {time.time()-t0:.1f}s, resp_chars={len(content)}")
                return content
            except (requests.ConnectionError, requests.Timeout) as e:
                last_err = e
                _log(f"role={role} model={model} NETWORK FAILED (attempt {attempt+1}): {e}")
                if attempt == 0:
                    time.sleep(2.0)   # дать Ollama опомниться после обрыва
                    break  # повтор внешнего цикла; на втором попытка не дублируется
                continue
            except requests.HTTPError as e:
                # 5xx у Ollama — временные сбои (OOM при загрузке модели, рестарт сервиса); ретраим
                status = getattr(getattr(e, "response", None), "status_code", 0) or 0
                last_err = e
                transient = 500 <= status < 600
                _log(f"role={role} model={model} HTTP {status or '?'} FAILED"
                     f"{' (transient)' if transient else ''}: {e}")
                if transient and attempt == 0:
                    time.sleep(3.0)
                    break  # повтор внешнего цикла с той же моделью, без перехода на fallback
                if i + 1 < len(models_to_try):
                    unload_model(model)
                continue
            except Exception as e:
                last_err = e
                _log(f"role={role} model={model} FAILED: {e}")
                # перед переходом на fallback-модель освобождаем VRAM явной выгрузкой
                if i + 1 < len(models_to_try):
                    unload_model(model)
                continue
        else:
            break  # внутренний цикл отработал без early-break по сети — выходим
    raise LLMError(f"Все модели роли '{role}' исчерпаны. Последняя ошибка: {last_err}")


def _wants_json(messages: list[dict]) -> bool:
    """JSON-режим включается явным маркером в system-промпте (плюс старые промпты — СК)."""
    fmt = ""
    for m in messages:
        if m["role"] == "system":
            fmt = m["content"]
    if JSON_FORMAT_MARKER in fmt:
        return True
    return ('"files"' in fmt or '"tasks"' in fmt or '"passed"' in fmt
            or '"new_tasks"' in fmt or '"gaps"' in fmt)


def _balanced_slice(text: str, start: int, opener: str, closer: str) -> str | None:
    """Возвращает сбалансированную по скобкам подстроку, игнорируя строки и экранирование."""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _repair_json(s: str) -> str:
    """Пытается починить типичные огрехи LLM-JSON: управляющие символы внутри строк."""
    out, in_str, esc = [], False, False
    for ch in s:
        if in_str:
            if esc:
                esc = False
                out.append(ch)
                continue
            if ch == "\\":
                esc = True
                out.append(ch)
                continue
            if ch == '"':
                in_str = False
                out.append(ch)
                continue
            if ch == "\n":
                out.append("\\n")   # литеральный перевод строки внутри "..."
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            if ch == "\r":
                continue
            out.append(ch)
        else:
            if ch == '"':
                in_str = True
            out.append(ch)
    return "".join(out)


def _loads_maybe_repaired(candidate: str):
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return json.loads(_repair_json(candidate))


def _looks_like_file_obj(o) -> bool:
    return isinstance(o, dict) and "path" in o and "content" in o


def _find_wrapped_obj(cleaned: str):
    """Возвращает {...}-объект с обёрточными ключами (files/tasks/...), пропуская
    вложенные объекты файлов вида {"path":...,"content":...}."""
    WRAP_KEYS = {"files", "tasks", "new_tasks", "plan_md", "passed",
                 "summary", "verdict", "notes"}
    start = cleaned.find("{")
    while start != -1:
        slice_ = _balanced_slice(cleaned, start, "{", "}")
        if slice_ is not None:
            try:
                obj = _loads_maybe_repaired(slice_)
            except Exception:
                obj = None
            if isinstance(obj, dict) and (set(obj) & WRAP_KEYS or not _looks_like_file_obj(obj)):
                return obj, start
        nxt = cleaned.find("{", start + 1)
        if nxt == start:
            break
        start = nxt
    return None, -1


def _truncate_to_last_complete_obj(text: str):
    """Для обрезанного max_tokens массива файлов: берём все полные {...}-элементы."""
    objs = []
    i = 0
    while True:
        start = text.find("{", i)
        if start == -1:
            break
        slice_ = _balanced_slice(text, start, "{", "}")
        if slice_ is None:
            break
        try:
            objs.append(_loads_maybe_repaired(slice_))
        except Exception:
            pass
        i = start + len(slice_)
    if objs and all(isinstance(o, dict) and o.get("path") for o in objs):
        # один полный файл -> {"files":[obj]} (roles.coder_generate сам поднимет одиночный объект);
        # несколько -> список, который roles принимают как голый массив файлов
        return {"files": objs, "notes": "(ответ был обрезан, взяты полные файлы)"}
    return None


def _salvage_plan_truncation(cleaned: str):
    """Спасение ОБРЕЗАННОГО планового JSON {"plan_md": "...", "tasks": [...]}.

    Типичный кейс: модель начинает отвечать с huge plan_md (структура проекта в
    markdown с ```-блоками), упирается в num_predict и не доходит до "tasks".
    Если tasks уже частично есть — берём полные элементы; иначе возвращаем
    plan-объект без задач (roles попросит повторить коротким ответом)."""
    pstart = cleaned.find('"plan_md"')
    tstart = cleaned.find('"tasks"')
    if pstart == -1 or tstart == -1 or tstart < pstart:
        return None
    # если после "tasks" нет '[' — массив ещё не начался
    obr = cleaned.find("[", tstart)
    if obr == -1:
        return None
    arr = _balanced_slice(cleaned, obr, "[", "]")
    if arr is not None:
        try:
            parsed = _loads_maybe_repaired(arr)
        except Exception:
            parsed = None
        if isinstance(parsed, list) and parsed and all(isinstance(x, dict) for x in parsed):
            return {"tasks": parsed}
    # оборванный массив: собираем только полные {...}-элементы задач
    objs = []
    i = obr + 1
    while True:
        s = cleaned.find("{", i)
        if s == -1:
            break
        sl = _balanced_slice(cleaned, s, "{", "}")
        if sl is None:
            break
        try:
            o = _loads_maybe_repaired(sl)
            if isinstance(o, dict) and ("title" in o or "id" in o):
                objs.append(o)
        except Exception:
            pass
        i = s + len(sl)
    if objs:
        return {"tasks": objs}
    # задач нет вообще — это всё ещё валидный частичный ответ (план есть),
    # но НЕ годится как план; помечаем, чтобы chat_json сделал укороченный ретрай
    return None


def _close_truncated_json(cleaned: str):
    """Автозакрытие обрезанного JSON: режем до последней полной строки/элемента и
    закрываем открытые { / [ . Лечит кейс planner, упёршегося в num_predict
    посреди длинного планового текста."""
    start = cleaned.find("{")
    if start == -1:
        return None
    frag = cleaned[start:]
    stack, in_str, esc = [], False, False
    last_safe, last_stack = 0, []
    for i, ch in enumerate(frag):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack and ((ch == "}" and stack[-1] == "{") or (ch == "]" and stack[-1] == "[")):
                stack.pop()
                last_safe, last_stack = i + 1, list(stack)
    # конец после полной вершины (глубина 0) — это был валидный объект, его уже нашли выше
    if not stack:
        return None
    cut = frag[:last_safe].rstrip().rstrip(",")
    tail = "".join("]" if c == "[" else "}" for c in reversed(last_stack))
    try:
        obj = _loads_maybe_repaired(cut + tail)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def extract_json(text: str):
    """Достаёт первый валидный JSON-объект/массив из ответа модели.

    Устойчив к: markdown-обёрткам, пояснениям до/после JSON, литеральным \\n
    внутри строк кода, обрезанному ответу (заканчиваемся на ...)."""
    cleaned = re.sub(r"```(json|python)?", "", text)

    # -1) приоритет объекту-обёртке {"plan_md"/"files"/"tasks"...}: если в тексте
    #     есть JSON-ключи вида "ключ":, а не только markdown-списки, пробуем его
    #     ДО поиска первого '[' (в plan_md часто встречается "[" как текст)
    wrapped0, wpos0 = _find_wrapped_obj(cleaned)
    if wrapped0 is not None and wrapped0.get("tasks"):
        return wrapped0

    # -2) обрезанный плановый ответ: {"plan_md": "...", "tasks": [ ... — сначала
    #     спасаем полные элементы tasks, иначе ниже одиночная полная задача внутри
    #     оборванного массива ложно считается целым ответом (регрессия qwen2.5 planner)
    if '"plan_md"' in cleaned and '"tasks"' in cleaned:
        salvaged0 = _salvage_plan_truncation(cleaned)
        if salvaged0 is not None:
            return salvaged0

    def _try_array_at(pos: int):
        """Если с позиции pos начинается массив файлов/задач — вернуть его."""
        if pos == -1:
            return None
        obj = _balanced_slice(cleaned, pos, "[", "]")
        if obj is None:
            return None
        try:
            parsed = _loads_maybe_repaired(obj)
        except Exception:
            return None
        if (isinstance(parsed, list) and parsed
                and all(isinstance(x, dict) for x in parsed)
                and all(_looks_like_file_obj(x) or "id" in x or "title" in x for x in parsed)):
            return parsed
        return None

    # 0) объект-обёртка {"files":...}/{"tasks":...} идёт в приоритете над голым массивом,
    #    если он встречается РАНЬШЕ первого '[' (иначе это markdown-список внутри текста)
    wrapped, wpos = _find_wrapped_obj(cleaned)
    apos = cleaned.find("[")
    # 1) голый массив [{"path":...},{"path":...}] — типичный ответ qwen3.5 на coder-запросе
    arr = _try_array_at(apos)
    if arr is not None and (wrapped is None or apos < wpos):
        return arr
    # 1b) массив оборван лимитом токенов (нет закрывающего ']') — собираем полные элементы;
    #     иначе ниже по коду одиночный полный {...} внутри него ложно считается целым ответом
    if apos != -1 and _balanced_slice(cleaned, apos, "[", "]") is None:
        trimmed0 = _truncate_to_last_complete_obj(cleaned[apos:])
        if trimmed0 is not None:
            return trimmed0
    # 2) объект-обёртка {"files":...}/{"/tasks":...}/{"passed":...}
    if wrapped is not None:
        return wrapped
    # 3) массив мог быть не распознан по первому '[' — пробуем следующие
    while True:
        nxt = cleaned.find("[", apos + 1)
        if nxt == -1 or nxt == apos:
            break
        apos = nxt
        arr = _try_array_at(apos)
        if arr is not None:
            return arr
    # 4) любой валидный {...} как есть; но если это одиночный объект файла внутри
    #    обрезанного {"files":[... — поднимаем его в корректную обёртку
    start = cleaned.find("{")
    while start != -1:
        slice_ = _balanced_slice(cleaned, start, "{", "}")
        if slice_ is not None:
            try:
                obj = _loads_maybe_repaired(slice_)
            except Exception:
                obj = None   # объект неполный (ответ обрезан) — идём дальше/к recovery
            if isinstance(obj, dict):
                if _looks_like_file_obj(obj) and '"files"' in cleaned[:start]:
                    return {"files": [obj], "notes": "(ответ был обрезан, взят полный файл)"}
                return obj
        nxt = cleaned.find("{", start + 1)
        if nxt == start:
            break
        start = nxt
    # 4b) ответ ОБРЕЗАН посреди незакрытой JSON-строки (частый кейс planner с длинным
    #     plan_md): обрезаем до последней полной строки и закрываем открытые скобки.
    #     Если при этом успел начаться массив tasks — спасаем полные элементы задач.
    salvaged = _salvage_plan_truncation(cleaned)
    if salvaged is not None:
        return salvaged
    closed = _close_truncated_json(cleaned)
    if closed is not None and isinstance(closed, dict):
        # объект-обёртка без "tasks" может быть ложным вложением внутри незакрытой
        # JSON-строки (например, фрагмент дерева файлов из plan_md) — не считаем его ответом
        if closed.get("tasks") or closed.get("files") or closed.get("new_tasks"):
            return closed
        wrapped_c, _ = _find_wrapped_obj(_repair_json(cleaned))
        if wrapped_c is not None:
            return wrapped_c
    # хвостовой шанс: JSON оборван лимитом токенов — собираем полные элементы
    trimmed = _truncate_to_last_complete_obj(cleaned)
    if trimmed is not None:
        return trimmed
    raise LLMError(f"В ответе модели не найден JSON:\n{text[:500]}")
