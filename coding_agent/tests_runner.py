"""Реальные проверки: компиляция Python, pytest, smoke-запуск. Никаких суждений на глаз."""
import ast
import os
import re
import subprocess
import sys

from . import config

PROJECT_CWD = None  # задаётся оркестратором (каталог проекта)

# импорт-модуль -> pip-пакет (для автоустановки зависимостей из requirements.txt)
_IMPORT_TO_PKG = {"flask": "flask", "fastapi": "fastapi", "uvicorn": "uvicorn",
                  "django": "django", "requests": "requests", "pytest": "pytest",
                  "sqlalchemy": "sqlalchemy", "pydantic": "pydantic", "jinja2": "jinja2"}


def _safe_name(s: str) -> bool:
    """Разрешаем только простые pip-имена (буквы/цифры/._-), без версий и флагов."""
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,50}", s))


def ensure_requirements_installed() -> None:
    """Если в проекте есть requirements.txt — ставим отсутствующие пакеты (один раз за сессию).
    Опасные строки (-r, git+, URL, версии) игнорируются; установка не блокирует процесс при ошибке."""
    global _REQ_DONE
    if _REQ_DONE or not config.AUTO_INSTALL_REQS:
        return
    _REQ_DONE = True
    req = os.path.join(_cwd(), "requirements.txt")
    if not os.path.isfile(req):
        return
    try:
        lines = open(req, encoding="utf-8").read().splitlines()
    except OSError:
        return
    to_install = []
    for ln in lines:
        name = re.split(r"[=<>!~;\[#]", ln.strip())[0].strip()
        if not name or name.startswith("#") or not _safe_name(name):
            continue
        pkg = _IMPORT_TO_PKG.get(name.lower().replace("_", "-"), name)
        probe = name.replace("-", "_")
        try:
            import importlib.util
            if importlib.util.find_spec(probe) is not None:
                continue
        except (ValueError, ModuleNotFoundError, AttributeError):
            pass
        to_install.append(pkg)
    if not to_install:
        return
    print(f"[deps] устанавливаю: {' '.join(to_install)}", flush=True)
    pip_base = [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check"]
    rc, _ = run([*pip_base, *to_install], timeout=config.PIP_TIMEOUT)
    if rc != 0:
        # отдельно пробуем каждый пакет (один битый не должен тащить вниз остальные)
        for pkg in to_install:
            rc1, _ = run([*pip_base, pkg], timeout=config.PIP_TIMEOUT)
            if rc1 != 0:
                print(f"[deps] НЕ удалось установить {pkg} — задача может упасть в проверках",
                      flush=True)


_REQ_DONE = False

# что модели-кодеру нельзя исполнять при smoke-проверке (side effects вне песочницы)
_DANGEROUS_CALLS = {
    "eval", "exec", "compile", "__import__", "open",        # builtins
}
_DANGEROUS_ATTRS = {
    ("os", "system"), ("os", "popen"), ("os", "remove"), ("os", "rmdir"),
    ("os", "unlink"), ("shutil", "rmtree"), ("subprocess", "run"),
    ("subprocess", "Popen"), ("subprocess", "call"), ("subprocess", "check_output"),
    ("socket", "socket"), ("ctypes", "CDLL"), ("pathlib", "Path"),
}
# сетевые bind/serve: чат-приложения на FastAPI/Flask часто зовут сервер прямо в smoke-проверке —
# это блокирует процесс до TEST_TIMEOUT и занимает порт. Разрешаем только создание приложения,
# запуск сервера остаётся на совести пользователя (инструкция в README).
_SERVER_ATTRS = {"bind", "serve", "serve_forever", "listen", "run_app", "run_simple"}
_DANGEROUS_ATTRS.update({("uvicorn", "run"), ("flask", "run")})


def _is_main_guard(node: ast.AST) -> bool:
    """True для узла if __name__ == '__main__' (и вариантов '=='/`== '__main__'`)."""
    return (isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Name) and node.left.id == "__name__"
            and any(isinstance(c, ast.Constant) and c.value == "__main__"
                    for c in node.comparators))


def scan_dangerous(src: str) -> list[str]:
    """AST-анализ: возвращает список опасных выражений (пусто => файл безопасен для исполнения)."""
    problems = []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return problems  # синтаксис ловит py_compile, здесь не наша забота
    # узлы внутри if __name__ == "__main__": при smoke-импорте (run_name='__notmain__')
    # не исполняются — сервер из main-guard не блокирует проверку и не занимает порт
    guarded_ids = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_main_guard(node.test):
            guarded_ids.update(id(ch) for ch in ast.walk(node))
    for node in ast.walk(tree):
        if id(node) in guarded_ids and not (isinstance(node, ast.Compare) and _is_main_guard(node)):
            continue   # опасный вызов под защитой main — не считаем
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in _DANGEROUS_CALLS:
                problems.append(f"строка {node.lineno}: вызов {fn.id}()")
            elif isinstance(fn, ast.Attribute):
                base = fn.value.id if isinstance(fn.value, ast.Name) else ""
                pair = (base, fn.attr)
                if pair in _DANGEROUS_ATTRS:
                    problems.append(f"строка {node.lineno}: вызов {base}.{fn.attr}()")
                elif fn.attr in {"system", "popen", "rmtree", "Popen"} or fn.attr in _SERVER_ATTRS:
                    problems.append(f"строка {node.lineno}: вызов ?.{fn.attr}()")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = ([a.name for a in node.names] if isinstance(node, ast.Import)
                    else [node.module or ""])
            for m in mods:
                root_m = (m or "").split(".")[0]
                if root_m in {"subprocess", "socket", "ctypes", "shutil"}:
                    problems.append(f"строка {node.lineno}: импорт {root_m}")
    return problems


def _cwd(cwd: str | None = None) -> str:
    return cwd or PROJECT_CWD or "."


def run(cmd: list[str], cwd: str | None = None, extra_args: list[str] | None = None,
        timeout: int | None = None) -> tuple[int, str]:
    try:
        p = subprocess.run([*cmd, *(extra_args or [])], cwd=_cwd(cwd), capture_output=True, text=True,
                           timeout=timeout or config.TEST_TIMEOUT)
        out = (p.stdout or "") + "\n" + (p.stderr or "")
        return p.returncode, out.strip()[-4000:]   # трейкаем вывод под бюджет контекста
    except subprocess.TimeoutExpired:
        return 124, f"TIMEOUT > {config.TEST_TIMEOUT}s: {' '.join(cmd)}"
    except FileNotFoundError as e:
        return 127, f"COMMAND NOT FOUND: {e}"


def py_compile_check(files: list[str]) -> tuple[bool, str]:
    """Синтаксис всех написанных .py файлов."""
    targets = [f for f in files if f.endswith(".py") and os.path.exists(os.path.join(_cwd(), f))]
    if not targets:
        return True, "(нет python-файлов)"
    # py_compile запускаем из корня проекта (run уже использует _cwd по умолчанию)
    rc, out = run([sys.executable, "-m", "py_compile", *targets])
    return rc == 0, out or "OK"


def pytest_available() -> bool:
    """Проверяем, что pytest реально запускается (окружение может ломать его плагины)."""
    rc, _ = run([sys.executable, "-m", "pytest", "--version"])
    return rc == 0


_PYTEST_OK: bool | None = None


def _run_pytest_files(files: list[str]) -> tuple[bool, str]:
    """pytest по конкретным файлам с -p no:cacheprovider (обход сломанных плагинов)."""
    rc, out = run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider",
                   "-x", "-q", "--maxfail=3", *files])
    if rc != 0 and ("_pytest/" in out or "pluggy" in out) and "passed" not in out:
        return False, "__PLUGIN_BROKEN__"
    return rc == 0, out


def _unittest_files(files: list[str]) -> tuple[bool, str | None]:
    """Прямой запуск test-файлов через unittest.main (работает без пакетов/__init__.py):
    python <file> -v. Файл сам вызывает unittest.main, если есть if __name__ == '__main__';
    иначе оборачиваем в runpy с discover-подобным прогоном."""
    outs = []
    overall_ok = True
    ran_any = False
    for f in files:
        full = os.path.join(_cwd(), f)
        if not os.path.exists(full):
            continue
        code = ("import unittest, sys, os\n"
                "sys.dont_write_bytecode=True\n"
                "os.chdir(sys.argv[1])\n"
                "sys.path.insert(0, os.getcwd())\n"
                "loader = unittest.TestLoader()\n"
                f"suite = loader.discover(os.path.dirname({f!r}) or '.', pattern='test_*.py', top_level_dir='.')\n"
                "runner = unittest.TextTestRunner(verbosity=2)\n"
                "res = runner.run(suite)\n"
                "sys.exit(0 if res.wasSuccessful() else 1)\n")
        rc, out = run([sys.executable, "-c", code], extra_args=[os.path.abspath(_cwd())])
        if "Ran 0 tests" in out:
            continue
        ran_any = True
        outs.append(f"--- {f} ---\n{out}")
        overall_ok &= rc == 0
    if not ran_any:
        return True, None
    return overall_ok, "\n".join(outs)[:4000]


def pytest_run(task_files: list[str] | None = None, cwd: str | None = None) -> tuple[bool, str] | None:
    """Прогон тестов по файлам задачи; для не-тестовых файлов — unittest-discover + pytest overall.
    Возвращает None только если тестов в проекте нет вообще."""
    test_files = [f for f in (task_files or [])
                  if os.path.basename(f).startswith("test_") and f.endswith(".py")]
    other_new = [f for f in (task_files or []) if f not in test_files]

    global _PYTEST_OK
    if _PYTEST_OK is None:
        _PYTEST_OK = pytest_available()

    results = []
    if test_files and _PYTEST_OK:
        ok, out = _run_pytest_files(test_files)
        if out == "__PLUGIN_BROKEN__":
            # pytest сломан окружением — пробуем unittest напрямую по тем же файлам
            u_ok, u_out = _unittest_files(test_files)
            if u_out is not None:
                results.append((u_ok, f"[unittest files {','.join(test_files)}]\n{u_out}"))
        else:
            results.append((ok, f"[pytest {','.join(test_files)}]\n{out}"))
    elif test_files and not _PYTEST_OK:
        u_ok, u_out = _unittest_files(test_files)
        if u_out is not None:
            results.append((u_ok, f"[unittest files {','.join(test_files)}]\n{u_out}"))

    cwd = _cwd(cwd)
    def _find_test_file() -> bool:
        skip = {".git", "__pycache__", "node_modules", ".venv", "venv", ".pytest_cache", "agent_state"}
        for dirpath, dirs, names in os.walk(cwd):
            dirs[:] = [d for d in dirs if d not in skip]
            for name in names:
                if name.startswith("test_") and name.endswith(".py"):
                    return True
        return False

    has_any_tests = bool(test_files) or _find_test_file() or os.path.isdir(os.path.join(cwd, "tests"))

    if other_new or (not test_files and has_any_tests):
        code = ("import unittest, sys, os\n"
                "sys.dont_write_bytecode=True\n"
                "os.chdir(sys.argv[1])\n"
                "sys.path.insert(0, os.getcwd())\n"
                "suite = unittest.TestLoader().discover('.', pattern='test_*.py', top_level_dir='.')\n"
                "res = unittest.TextTestRunner(verbosity=2).run(suite)\n"
                "sys.exit(0 if res.wasSuccessful() else 1)\n")
        rc, uout = run([sys.executable, "-c", code], extra_args=[os.path.abspath(cwd)])
        no_tests_found = ("Ran 0 tests" in uout or "NO TESTS" in uout.upper())
        if not (no_tests_found and rc != 0):
            # пустой/не найденный прогон — не провал проекта (реальный сигнал даёт pytest overall)
            results.append((rc == 0, f"[unittest discover]\n{uout}"))
        if _PYTEST_OK:
            pok, pout = _run_pytest_files([])   # весь проект
            if pout != "__PLUGIN_BROKEN__":
                results.append((pok, f"[pytest overall]\n{pout}"))

    if not results:
        return None
    final_ok = all(ok for ok, _ in results)
    return final_ok, "\n\n".join(txt for _, txt in results)


def smoke_import_check(files: list[str]) -> tuple[bool, str]:
    """Каждый новый .py исполняется как модуль (runpy) — ловим ошибки импорта/синтаксиса
    без требования к тому, что tests/ или корень проекта являются пакетами.
    Файлы с опасными вызовами (os.system/subprocess/eval...) НЕ исполняются — только компилируются."""
    problems = []
    for f in files:
        full = os.path.join(_cwd(), f)
        if not (f.endswith(".py") and os.path.exists(full)):
            continue
        try:
            src = open(full, encoding="utf-8").read()
        except (OSError, UnicodeDecodeError):
            continue
        dangers = scan_dangerous(src)
        if dangers:
            # не запускаем код с side effects — фиксируем как предупреждение, но не FAIL:
            # легальный сервер может использовать subprocess; решение за тестером-моделью
            problems.append(f"{f}: [WARN] не исполнялся (опасные вызовы: {'; '.join(dangers[:5])})")
            continue
        code = ("import runpy, sys, os\n"
                "sys.dont_write_bytecode=True\n"
                "os.chdir(sys.argv[1])\n"
                "sys.path.insert(0, os.getcwd())\n"
                f"runpy.run_path({f!r}, run_name='__notmain__')\nprint('EVAL OK')\n")
        rc, out = run([sys.executable, "-c", code], extra_args=[os.path.abspath(_cwd())])
        if rc != 0:
            problems.append(f"{f}: {out}")
    real_fail = [p for p in problems if "[WARN]" not in p]
    if real_fail:
        return False, "\n".join(real_fail)[:3000]
    if problems:
        return True, "\n".join(problems)[:2000]   # предупреждения уходят в вывод тестеру
    return True, "(smoke-eval OK)"


def verify_written(task_files: list[str]) -> tuple[bool, str]:
    """Полный конвейер проверок после записи файлов задачи."""
    parts = []
    ok = True
    ensure_requirements_installed()   # если кодер создал requirements.txt — ставим зависимости
    cok, cout = py_compile_check(task_files)
    parts.append(f"[py_compile rc={'ok' if cok else 'FAIL'}]\n{cout}")
    ok &= cok
    simport_ok, simport_out = smoke_import_check(task_files)
    parts.append(f"[imports rc={'ok' if simport_ok else 'FAIL'}]\n{simport_out}")
    ok &= simport_ok
    pr = pytest_run(task_files)
    if pr is not None:
        pok, pout = pr
        parts.append(f"[pytest rc={'ok' if pok else 'FAIL'}]\n{pout}")
        ok &= pok
    else:
        parts.append("[pytest] тестов пока нет")
    return ok, "\n\n".join(parts)
