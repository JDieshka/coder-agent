"""Конфигурация кодера-агента: модели, бюджет контекста, роли."""
from dataclasses import dataclass

OLLAMA_URL = "http://localhost:11434"

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
CTX_WINDOW = 16384
MAX_PROMPT_TOKENS = 11000     # резерв под completion и KV-кэш
KEEP_ALIVE = "0"              # немедленная выгрузка модели -> ротация ролей без OOM

MAX_DEBUG_ROUNDS = 3          # циклов coder->tester на одну задачу
TEST_TIMEOUT = 120            # секунд на запуск проверок
REQ_TIMEOUT = 600             # секунд на HTTP-запрос к ollama

STATE_DIR = "agent_state"
PLAN_FILE = f"{STATE_DIR}/plan.md"
TASKS_FILE = f"{STATE_DIR}/tasks.json"
PROGRESS_FILE = f"{STATE_DIR}/progress.md"
AUDIT_FILE = f"{STATE_DIR}/audit_report.md"
LOG_FILE = f"{STATE_DIR}/run.log"
