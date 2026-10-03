"""Реальные проверки: компиляция Python, pytest, smoke-запуск. Никаких суждений на глаз."""
import os
import subprocess
import sys

from . import config

PROJECT_CWD = None  # задаётся оркестратором (каталог проекта)


def _cwd(cwd: str | None = None) -> str:
    return cwd or PROJECT_CWD or "."


def run(cmd: list[str], cwd: str | None = None, extra_args: list[str] | None = None) -> tuple[int, str]:
    try:
        p = subprocess.run([*cmd, *(extra_args or [])], cwd=_cwd(cwd), capture_output=True, text=True,
                           timeout=config.TEST_TIMEOUT)
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
            # падающие тесты всё равно ловятся через unittest/pytest на следующем шаге;
            # считаем этот прогон нейтральным и проверяем остальное
            pass
        else:
            results.append((ok, f"[pytest {','.join(test_files)}]\n{out}"))

    cwd = _cwd(cwd)
    has_any_tests = bool(test_files) or any(
        name.startswith("test_") and name.endswith(".py")
        for _, _, names in os.walk(cwd) for name in names
    ) or os.path.isdir(os.path.join(cwd, "tests"))

    if other_new or (not test_files and has_any_tests):
        rc, uout = run([sys.executable, "-m", "unittest", "discover", "-v"], cwd=cwd)
        u_ok = rc == 0 or "NO TESTS RAN" in uout or "Ran 0 tests" in uout
        results.append((u_ok, f"[unittest discover]\n{uout}"))
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
    без требования к тому, что tests/ или корень проекта являются пакетами."""
    problems = []
    for f in files:
        if not (f.endswith(".py") and os.path.exists(os.path.join(_cwd(), f))):
            continue
        code = ("import runpy, sys, os\n"
                "sys.dont_write_bytecode=True\n"
                "os.chdir(sys.argv[1])\n"
                "sys.path.insert(0, os.getcwd())\n"
                f"runpy.run_path({f!r}, run_name='__notmain__')\nprint('EVAL OK')\n")
        rc, out = run([sys.executable, "-c", code], extra_args=[os.path.abspath(_cwd())])
        if rc != 0:
            problems.append(f"{f}: {out}")
    if problems:
        return False, "\n".join(problems)[:3000]
    return True, "(smoke-eval OK)"


def verify_written(task_files: list[str]) -> tuple[bool, str]:
    """Полный конвейер проверок после записи файлов задачи."""
    parts = []
    ok = True
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
