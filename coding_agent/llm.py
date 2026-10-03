"""Клиент Ollama: вызовы с ротацией моделей (keep_alive=0) и парсингом JSON."""
import json
import re
import time

import requests

from . import config


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


def chat(role: str, messages: list[dict]) -> str:
    """Один «автоматическая сессия»: загрузить модель -> запрос -> выгрузить."""
    rc = config.ROLES[role]
    models_to_try = [rc.model] + ([rc.fallback_model] if rc.fallback_model else [])
    last_err = None
    for model in models_to_try:
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
        except Exception as e:
            last_err = e
            _log(f"role={role} model={model} FAILED: {e}")
            # fallback на другую модель только при ошибке загрузки/OOM
            continue
    raise LLMError(f"Все модели роли '{role}' исчерпаны. Последняя ошибка: {last_err}")


def extract_json(text: str):
    """Достаёт первый валидный JSON-объект/массив из ответа модели."""
    text = re.sub(r"```(json)?", "", text)
    # жёсткий поиск балансной {...} или [...]
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        while start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == opener:
                    depth += 1
                elif text[i] == closer:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break
            start = text.find(opener, start + 1)
    # последний шанс: eval-подобный repair трюк с кавычками
    raise LLMError(f"В ответе модели не найден JSON:\n{text[:500]}")
