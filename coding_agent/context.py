"""Контекст проекта: карта файлов, чтение/запись кода, бюджет промпта под 16K."""
import ast
import os

from . import config

IGNORE_DIRS = {".git", "agent_state", "__pycache__", "node_modules", ".venv", "venv",
               ".pytest_cache", "dist", "build"}

# точный токенизатор (опционально): если tiktoken не установлен — остаётся грубая эвристика
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")   # близко к чату Qwen; консервативная оценка
except Exception:
    _ENC = None


def token_cost(s: str) -> int:
    """Стоимость строки в токенах: tiktoken (точно) или ~3.8 символа/токен (эвристика)."""
    if not s:
        return 0
    if _ENC is not None:
        try:
            return len(_ENC.encode(s, disallowed_special=())) + 4  # запас на role/разделители
        except Exception:
            pass
    return int(len(s) / 3.8) + 1

# файлы, которые модели-кодеру запрещено трогать (состояние агента и служебное)
PROTECTED_FILES = {"requirements.txt", ".gitignore"}


def is_protected(path: str) -> bool:
    """True, если файл относится к состоянию агента/службе — писать его нельзя."""
    norm = path.replace("\\", "/").lstrip("./")
    parts = norm.split("/")
    if "agent_state" in parts or ".git" in parts:
        return True
    base = parts[-1]
    if base in PROTECTED_FILES or base.endswith((".log", ".lock")):
        return True
    if base == "request.txt":
        return True
    return False


def approx_tokens(s: str) -> int:
    """Совместимость: делегирует в token_cost (tiktoken, если доступен)."""
    return token_cost(s)


def _cut_to_tokens(text: str, max_tokens: int) -> str:
    """Обрезает строку не длиннее max_tokens токенов (бинарный поиск по символам)."""
    if token_cost(text) <= max_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if token_cost(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + "\n...[усечено из-за лимита контекста]"


def project_path(path: str) -> str:
    """Абсолютный/относительный путь файла проекта от корня проекта."""
    return os.path.join(config.PROJECT_DIR, path.replace("\\", "/"))


def project_map() -> str:
    root = config.PROJECT_DIR or "."
    lines = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        rel = os.path.relpath(dirpath, root).replace("\\", "/")
        for fn in sorted(filenames):
            if fn.endswith((".py", ".md", ".txt", ".json", ".html", ".css", ".js")):
                p = os.path.join(dirpath, fn)
                rp = f"{rel}/{fn}" if rel != "." else fn
                try:
                    src = open(p, encoding="utf-8").read()
                except (OSError, UnicodeDecodeError):
                    continue
                sig = _py_signature(src) if fn.endswith(".py") else ""
                size = len(src.splitlines())
                lines.append(f"{rp} ({size} строк){sig}")
    return "\n".join(lines) if lines else "(пусто)"


def _py_signature(src: str) -> str:
    """AST-сигнатуры: классы/функции — компактное представление интерфейса файла."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ""
    parts = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            parts.append(f" class {node.name}({','.join(methods)})")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = ", ".join(a.arg for a in node.args.args)
            parts.append(f" def {node.name}({args})")
    return " |" + ";".join(parts) if parts else ""


def read_file(path: str) -> str:
    try:
        with open(project_path(path), encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def write_file(path: str, content: str):
    if is_protected(path):
        raise PermissionError(f"защита состояния: файл '{path}' не может быть перезаписан моделью")
    full = project_path(path)
    # двойная страховка: итоговый путь обязан оставаться внутри каталога проекта
    root_abs = os.path.abspath(config.PROJECT_DIR or ".")
    if not os.path.abspath(full).startswith(root_abs + os.sep) and os.path.abspath(full) != root_abs:
        raise ValueError(f"путь вне каталога проекта: {path}")
    os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)


def build_messages(system: str, blocks: list[tuple[str, str]],
                   budget: int = config.MAX_PROMPT_TOKENS) -> list[dict]:
    """Собирает prompt из приоритетных блоков; при переполнении урезает/отбрасывает хвостовые.

    Счёт ведётся по ИТОГОВОЙ user-строке (с заголовками '### title'), бюджет — это
    system+user вместе. Если один системный промпт больше бюджета — он усeкается до
    80% бюджета, чтобы в промпт обязательно попали задача и карта проекта.
    """
    sys_cost = token_cost(system)
    if sys_cost > budget:
        system = _cut_to_tokens(system, int(budget * 0.8))
        sys_cost = token_cost(system)

    kept: list[tuple[str, str]] = []

    def joined(bs) -> str:
        return "\n\n".join(f"### {t}\n{c}" for t, c in bs)

    for title, text in blocks:
        trial = kept + [(title, text)]
        if sys_cost + token_cost(joined(trial)) <= budget:
            kept.append((title, text))
            continue
        # не влезает целиком: пробуем урезать этот блок до остатка бюджета.
        # считаем итог именно по собранной строке (token_cost конкатенации != сумме частей),
        # иначе при системном промпте ~budget первый блок никогда не проходил бы проверку
        remaining = max(0, budget - sys_cost - 60)
        cut = _cut_to_tokens(text, remaining)
        trial_cut = kept + [(title, cut)]
        if token_cost(f"### {title}\n") > remaining or \
                sys_cost + token_cost(joined(trial_cut)) > budget:
            trial_cut = kept          # усечённый блок тоже не влезает — отбрасываем его
        kept = trial_cut
        break
    return [{"role": "system", "content": system}, {"role": "user", "content": joined(kept)}]
