"""Контекст проекта: карта файлов, чтение/запись кода, бюджет промпта под 16K."""
import ast
import os

from . import config

IGNORE_DIRS = {".git", "agent_state", "__pycache__", "node_modules", ".venv", "venv",
               ".pytest_cache", "dist", "build"}


def approx_tokens(s: str) -> int:
    # грубая оценка: ~3.8 символа на токен для смешанных RU/EN+кода
    return int(len(s) / 3.8) + 1


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
    full = project_path(path)
    os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)


def build_messages(system: str, blocks: list[tuple[str, str]],
                   budget: int = config.MAX_PROMPT_TOKENS) -> list[dict]:
    """Собирает prompt из приоритетных блоков; при переполнении отбрасывает хвостовые блоки."""
    header_chars = sum(approx_tokens(t) for _, t in [("sys", system)])
    used = header_chars
    kept = []
    for title, text in blocks:
        cost = approx_tokens(text) + approx_tokens(title) + 4
        if used + cost > budget:
            # урезаем конкретный блок до остатка бюджета, если он важен
            remaining = budget - used
            if remaining > 300:
                cut = int(remaining * 3.8)
                text = text[:cut] + "\n...[усечено из-за лимита контекста]"
                kept.append((title, text))
            break
        kept.append((title, text))
        used += cost
    user = "\n\n".join(f"### {t}\n{c}" for t, c in kept)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
