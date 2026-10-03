"""Конфигурация кодера-агента: модели, бюджет контекста, роли, пути проектов."""
import os
import re
from dataclasses import dataclass

OLLAMA_URL = os.environ.get("CODER_AGENT_OLLAMA_URL", "http://localhost:11434")

# Модели, которые будут использоваться (должны быть в `ollama pull`).
MODEL_PLANNER = "qwen2.5-coder:7b-instruct-q4_K_M"   # пишет план и декомпозицию
MODEL_CODER   = "qwen3.5:9b-q4_K_M"                  # пишет код
MODEL_TESTER  = "qwen3:8b-q4_K_M"                    # тестер/дебаггер + финальная сверка с планом
MODEL_FALLBACK_CODER = "qwen2.5-coder:7b-instruct-q4_K_M"

@dataclass
class RoleConfig:
    name: str
    model: str
    fallback_model: str | None = None
    temperature: float = 0.2
    max_tokens: int = 2048
    think: bool = False   # включённый reasoning у qwen3

ROLES = {
    "planner": RoleConfig("planner", MODEL_PLANNER, temperature=0.3, max_tokens=2500),
    "coder":   RoleConfig("coder",   MODEL_CODER, fallback_model=MODEL_FALLBACK_CODER,
                          temperature=0.1, max_tokens=3000),
    "tester":  RoleConfig("tester",  MODEL_TESTER, temperature=0.2, max_tokens=2000, think=True),
    "auditor": RoleConfig("auditor", MODEL_TESTER, temperature=0.1, max_tokens=2000, think=True),
}

# Жёсткий бюджет контекста (RTX 4060 8GB + 16GB RAM).
# CTX_WINDOW переопределяется env CODER_AGENT_CTX (например, CODER_AGENT_CTX=8192
# при нехватке VRAM; MAX_PROMPT_TOKENS автоматически подстрачивается ниже).
CTX_WINDOW = int(os.environ.get("CODER_AGENT_CTX", "16384"))
MAX_PROMPT_TOKENS = min(int(os.environ.get("CODER_AGENT_MAX_TOKENS", "11000")),
                        max(512, CTX_WINDOW - 4000))  # резерв под completion и KV-кэш
KEEP_ALIVE = "0"              # немедленная выгрузка модели -> ротация ролей без OOM

MAX_DEBUG_ROUNDS = 3          # циклов coder->tester на одну задачу
TEST_TIMEOUT = 120            # секунд на запуск проверок
REQ_TIMEOUT = 600             # секунд на HTTP-запрос к ollama
PIP_TIMEOUT = 300             # секунд на pip install зависимостей проекта

# автоустановка зависимостей из requirements.txt проекта (pip install отсутствующих пакетов).
# Выключается переменной окружения CODER_AGENT_NO_INSTALL=1.
AUTO_INSTALL_REQS = os.environ.get("CODER_AGENT_NO_INSTALL", "0") != "1"

# Все проекты агента живут в одной папке workflow/<имя_проекта>/:
# там и код проекта, и его состояние (agent_state/).
WORKFLOW_ROOT = os.environ.get("CODER_AGENT_WORKFLOW", "workflow")

# Пути ниже вычисляются в init_paths() после выбора/создания проекта.
PROJECT_DIR = ""
STATE_DIR = ""
PLAN_FILE = ""
TASKS_FILE = ""
PROGRESS_FILE = ""
AUDIT_FILE = ""
LOG_FILE = ""
BAR_FILE = ""


def slugify(text: str, max_len: int = 40) -> str:
    """Латиница/кириллица -> безопасное имя папки (кириллица транслитерируется)."""
    translit = {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
        "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
        "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
        "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch",
        "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    }
    text = text.lower()
    out = []
    for ch in text:
        if ch in translit:
            ch = translit[ch]
        for c in ch:
            if c.isascii() and (c.isalnum() or c in "-_"):
                out.append(c)
            elif c in " .,:;!?/\\()":
                out.append("-")
    slug = re.sub(r"-{2,}", "-", "".join(out)).strip("-")[:max_len].strip("-")
    return slug or "project"


def make_project_id(base: str) -> str:
    """Уникальный id проекта: base, base-2, base-3 ... внутри WORKFLOW_ROOT."""
    candidate = base
    n = 2
    while os.path.exists(os.path.join(WORKFLOW_ROOT, candidate)):
        candidate = f"{base}-{n}"
        n += 1
    return candidate


def init_paths(project_dir: str):
    """Фиксирует рабочий каталог проекта и все пути состояния относительно него."""
    global PROJECT_DIR, STATE_DIR, PLAN_FILE, TASKS_FILE
    global PROGRESS_FILE, AUDIT_FILE, LOG_FILE, BAR_FILE
    PROJECT_DIR = project_dir
    STATE_DIR = os.path.join(PROJECT_DIR, "agent_state")
    PLAN_FILE = os.path.join(STATE_DIR, "plan.md")
    TASKS_FILE = os.path.join(STATE_DIR, "tasks.json")
    PROGRESS_FILE = os.path.join(STATE_DIR, "progress.md")
    AUDIT_FILE = os.path.join(STATE_DIR, "audit_report.md")
    LOG_FILE = os.path.join(STATE_DIR, "run.log")
    BAR_FILE = os.path.join(STATE_DIR, "bar.json")


def find_existing_project(request: str) -> str | None:
    """Если проект с таким же описанием уже есть — возвращаем его папку (resume)."""
    if not os.path.isdir(WORKFLOW_ROOT):
        return None
    for name in sorted(os.listdir(WORKFLOW_ROOT)):
        d = os.path.join(WORKFLOW_ROOT, name)
        rf = os.path.join(d, "agent_state", "request.txt")
        if os.path.isdir(d) and os.path.isfile(rf):
            try:
                if open(rf, encoding="utf-8").read().strip() == request.strip():
                    return d
            except OSError:
                continue
    return None
