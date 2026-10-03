# Промт для реализации кодера-агента (точное воспроизведение текущей системы)

Скопируй всё, что ниже этой строки, и отправь ассистенту/модели в новый пустой проект.
Задача — реализовать систему ровно так, как описано, без «своего видения» архитектуры.

---

## ТЗ: локальный кодер-агент на трёх моделях Ollama с автоматической ротацией ролей

Реализуй Python-пакет `coding_agent` — агент, который по текстовому описанию пользователя
сам пишет проект. Железо: RTX 4060 8 ГБ VRAM, 16 ГБ RAM, жёсткий лимит контекста модели —
**16384 токена**. Поэтому три модели одновременно в памяти жить не могут — нужна автоматическая
ротация (загрузил → выполнил роль → выгрузил).

### Модели (уже скачаны через `ollama pull`, имена фиксировать в конфиге)
- **Planner**: `qwen2.5-coder:7b-instruct-q4_K_M` — пишет план и декомпозицию, делает ре-план.
- **Coder**: `qwen3.5:9b-q4_K_M` — пишет код по одной задаче; fallback при OOM/ошибке: `qwen2.5-coder:7b-instruct-q4_K_M`.
- **Tester/Debugger + Auditor**: `qwen3:8b-q4_K_M` — оценивает acceptance по реальному выводу проверок; та же модель делает финальную сверку с планом.

### Структура проекта (ровно эти файлы)
```
coding_agent/
  __init__.py        # from .orchestrator import main
  __main__.py        # python -m coding_agent "запрос"
  config.py          # модели, роли, бюджеты, пути, slugify, resume-поиск
  llm.py             # клиент Ollama + устойчивый парсинг JSON
  roles.py           # системные промпты четырёх ролей + парсинг ответов
  context.py         # карта проекта (AST-сигнатуры), чтение/запись, бюджет промпта
  state.py           # plan.md / tasks.json / progress.md / bar.json (atomic writes)
  tests_runner.py    # РЕАЛЬНЫЕ проверки: py_compile, smoke-import, unittest/pytest
  orchestrator.py    # машина состояний задач, фазы, git-коммиты, CLI main()
  progress_bar.py    # live прогресс-бар в одну строку консоли
requirements.txt     # requests>=2.31
README.md            # описание + инструкция запуска/настройки/resume
```

### config.py
- `OLLAMA_URL = "http://localhost:11434"` (env `CODER_AGENT_OLLAMA_URL`).
- Dataclass `RoleConfig(name, model, fallback_model=None, temperature, max_tokens, think=False)`;
  словарь `ROLES`: planner(t=0.3, max_tokens=2500), coder(t=0.1, max_tokens=3000, fallback),
  tester(t=0.2, max_tokens=2000, think=True), auditor(как tester).
- `CTX_WINDOW = 16384`, `MAX_PROMPT_TOKENS = 11000` (резерв под completion/KV-кэш),
  `KEEP_ALIVE = "0"` (немедленная выгрузка модели — основа ротации),
  `MAX_DEBUG_ROUNDS = 3`, `TEST_TIMEOUT = 120`, `REQ_TIMEOUT = 600`.
- `WORKFLOW_ROOT = env CODER_AGENT_WORKFLOW или "workflow"` — ВСЕ проекты пишутся только туда,
  текущая папка остаётся чистой.
- `slugify(text, max_len=40)` — транслитерация кириллицы в безопасное имя папки
  («Напиши чат-приложение…» → `napishi-chat-prilozhenie-s-autentifikats`).
- `make_project_id(base)` — уникальность: `base`, `base-2`, `base-3`…
- Глобальные пути `PROJECT_DIR/STATE_DIR/PLAN_FILE/TASKS_FILE/PROGRESS_FILE/AUDIT_FILE/LOG_FILE/BAR_FILE`
  заполняются функцией `init_paths(project_dir)`; состояние живёт в `<проект>/agent_state/`.
- `find_existing_project(request)` —resume: обходит `workflow/*/agent_state/request.txt`,
  возвращает папку проекта с идентичным запросом.

### llm.py
- Класс `LLMError(RuntimeError)`.
- `list_models()` / `check_models()` — GET `/api/tags`, при отсутствии нужных моделей — понятная ошибка
  с подсказкой `ollama pull`.
- `chat(role, messages)` — POST `/api/chat`, `stream: False`, `keep_alive: "0"`,
  options: `temperature`, `num_ctx=16384`, `num_predict` из RoleConfig. Для qwen3-моделей при
  `think=False` передавать `"think": false` (экономия контекста). Если в system-промпте есть ключи
  `"files"/"tasks"/"new_tasks"/"passed"/"gaps"` — включать `"format": "json"`
  (grammar-constrained decoding). Логировать role/model/длину сообщения/время ответа. При ошибке
  (OOM/не загрузилась) пробовать `fallback_model`; всё исчерпано → `LLMError`.
- Устойчивый `extract_json(text)` — критически важно (реальные LLM-ответы грязные). Должен переживать:
  markdown-обёртки ```...```, текст до/после JSON, литеральные переводы строк внутри JSON-строк
  (функция `_repair_json` экранирует \n/\t), голый массив `[{"path":...,"content":...}]` вместо
  объекта-обёртки (частый ответ qwen3.5 — именно из-за этого был краш `AttributeError: 'list' object
  has no attribute 'get'`), одиночный объект файла вместо обёртки, обрезанный max_tokens ответ
  (`_truncate_to_last_complete_obj` собирает все полные {...}-элементы в {"files":[...]}).
  Вспомогательные: `_balanced_slice` (баланс скобок с учётом строк/экранирования),
  `_find_wrapped_obj` (приоритет объектов с обёрточными ключами files/tasks/… над вложенными
  file-объектами), `_coerce_dict` в roles.py приводит список к словарю.

### roles.py — системные промпты (сохранить смысл и формат дословно)
- `PLANNER_SYSTEM`: план на русском, минимальный стек, декомпозиция на 5–12 МАЛЕНЬКИХ задач
  (каждая — один-два файла, выполнима за один проход при контексте ≤16K); порядок: каркас →
  аутентификация → модель данных → личные чаты → групповые чаты → тесты. Ответ строго
  `{"plan_md": "...", "tasks": [{"id":1,"title":"...","goal":"...","files":["..."],"acceptance":"..."}]}`.
- `REPLAN_SYSTEM`: по плану+статусам+прогрессу предложить НОВЫЕ задачи с новыми id,
  `{"new_tasks":[...]}`, или пустой список.
- `CODER_SYSTEM`: пишет полный содержимый код без TODO-заглушек, согласуется с картой проекта,
  ответ строго `{"files":[{"path":"...","content":"..."}],"notes":"..."}` + 4 правила формата
  (никакого текста вокруг JSON; содержимое файла — одна JSON-строка с \\n; большой файл = один
  элемент; без trailing commas). `DEBUG_CODER_SYSTEM` = CODER_SYSTEM + указание исправить
  диагностику минимальными правками.
- `TESTER_SYSTEM`: `{\"passed\": true|false, \"diagnosis\": "..."}` — судит по РЕАЛЬНОМУ выводу проверок.
- `AUDITOR_SYSTEM`: `{\"summary\":..., \"covered\":[...], \"gaps\":[...]}` — сверка плана и факта.
- `chat_json(role, messages, parse, retries=2)` — при ошибке парсинга дописывает в диалог
  «Твой ответ не является корректным JSON… повтори строго в формате», повтор до 3 попыток.
- Функции: `planner_create`, `planner_replan`, `coder_generate` (нормализует files, требует непустой
  результат), `tester_judge` (терпит голый bool и вложенный verdict), `auditor_check`,
  `_normalize_tasks` (гарантирует поля id/title/goal/files/acceptance/status/debug_rounds).

### context.py
- `approx_tokens(s) = len(s)/3.8 + 1` (грубая оценка RU/EN+код).
- Все файловые операции — только относительно `config.PROJECT_DIR` (`project_path`, `read_file`, `write_file`).
- `project_map()` — обход дерева (игнор: .git, agent_state, __pycache__, node_modules, venv, …),
  для .py файлов добавляет AST-сигнатуры `class X(m1,m2) | def f(args)` — компактный интерфейс
  вместо полного кода (экономия 16K).
- `build_messages(system, blocks, budget=MAX_PROMPT_TOKENS)` — блоки приоритетные: ЦЕЛЬ(план ≤3000
  симв.), ЗАДАЧА, КАРТА ПРОЕКТА, СУЩЕСТВУЮЩИЙ СВЯЗАННЫЙ КОД, ПРОГРЕСС, ДИАГНОСТИКА; при переполнении
  бюджета хвостовые блоки отбрасываются/урезаются с пометкой «[усечено из-за лимита контекста]».

### state.py
- `ensure_state()`, `save_request()` (пишется ОДИН раз — по нему работает resume-поиск).
- Atomic write через `.tmp` + `os.replace` для tasks.json и bar.json.
- `load_progress_tail(n=40)` — последние строки прогресса в промпт (скользящее резюме вместо всей истории).
- Снапшот бара: `save_bar_snapshot/load_bar_snapshot`.

### tests_runner.py — принятие задачи только по реальным exit code
- `run(cmd)` — subprocess с `cwd=каталог проекта`, timeout TEST_TIMEOUT, вывод трейкается до 4000 симв.
- `verify_written(files)` = конвейер:
  1. `py_compile` всех написанных .py;
  2. smoke-import: каждый .py исполняется через `runpy.run_path(run_name='__notmain__')` (ловит ошибки импорта без требования пакетной структуры);
  3. тесты: pytest по файлам `test_*.py` с `-p no:cacheprovider -x -q --maxfail=3`; отдельно
     `unittest discover -v`; pytest overall по проекту. Отсутствие pytest в окружении НЕ считается
     провалом (проверять один раз кэшем `_PYTEST_OK`, ловить сломанные плагины маркером `__PLUGIN_BROKEN__`).
- Итог ok = все выполненные проверки прошли.

### orchestrator.py — ядро
- `_log(msg)` — timestamp + печать + запись в `agent_state/run.log`; перед логом очищать строку бара.
- `git(*args)` — все git-команды выполняются с `cwd=config.PROJECT_DIR` (у каждого проекта свой
  репозиторий; git не обязателен — ошибки глотать).
- `dependency_code_for(task, map_)` — код связанных существующих файлов в промпт кодера: фильтр по
  ключевым словам (app/main/db/models/auth/chat + первый файл задачи), максимум 6 файлов, бюджет ~5200 токенов.
- `_normalize_task(t)` — доп. поля по умолчанию (защита от KeyError 'status', когда модель не вернула status).
- Фаза PLANNER: если plan+tasks уже есть — пропуск (resume). Иначе planner_create → сохранить.
- Цикл задач (pending → in_progress):
  - coder_generate (map + dep_code + progress_tail + diagnosis); ошибка LLMError/пустой список файлов →
    debug_round += 1, новая диагностика «верни СТРОГО валидный JSON», при достижении MAX_DEBUG_ROUNDS →
    статус failed, переход к следующей задаче (НЕ падать!).
  - защита от path traversal («..», абсолютные пути — игнорировать);
  - verify_written → hard_ok + hard_out;
  - tester_judge по фрагментам кода (≤1500 симв/файл) и реальному выводу; ошибка формата тестера →
    довериться hard-проверкам (не ронять процесс);
  - accepted = hard_ok AND passed → done: append_progress, `git add -A; git commit -m "task #N: title"`,
    save; иначе rounds+=1, diagnosis = текст тестера + «РЕАЛЬНЫЕ ОШИБКИ:\n» + hard_out; лимит раундов → failed.
- Фаза РЕ-ПЛАН: после основного цикла; failed-задачи получают второй шанс (status=pending, rounds=0);
  новые задачи добавляются только с непересекающимися id; ошибка реплана — пропускаем, идём к сверке.
  Затем повтор цикла задач.
- Фаза АУДИТ: auditor_check по плану/статусам/карте/прогрессу → `audit_report.md`
  (Итог/Реализовано/Пробелы); при ошибке формата — механическая сверка по статусам. Если gaps есть —
  ещё один цикл реплан→задачи→аудит.
- `run()` оборачивает весь конвейер в try/except LLMError: при недоступной Ollama/OOM — вернуть
  in_progress-задачи в pending, сохранить состояние, поднять ошибку с сообщением «продолжите той же
  командой (resume)».
- `main()` CLI: `python -m coding_agent "описание проекта" [--name my-app] [--fresh]`;
  выбор проекта: resume по запросу → resume по имени → новый `workflow/<slug>/`; `--fresh` удаляет
  plan/tasks/progress/audit; создаёт папки, `git init` + init-коммит; печатает итоговый JSON
  `{"done":N,"total":M,"gaps":K}`.

### progress_bar.py — live-строка без зависимостей
- Класс ProgressBar: `\r`-перерисовка, печать только при `sys.stdout.isatty()` и
  `CODER_AGENT_NO_BAR != "1"`.
- Формат: `[ 33%] ████████░░░░ 2/6 | Заголовок | КОД | ⏱ 4м12с ETA 8м30с`; ширина под терминал
  (shutil.get_terminal_size), бар ≥8 символов; проценты по ОБРАБОТАННЫМ (done+failed); ETA по средней
  скорости на задачу; фаза: ПЛАН/КОД/ОТЛАДКА(n/3)/РЕ-ПЛАН/СВЕРКА.
- Методы: set_plan/start_task/debug_round/finish_task/set_phase/mark_failed/clear/print_final,
  snapshot()/restore() (restore сдвигает `_start_ts` так, чтобы elapsed продолжился из снапшота);
  снапшот хранится в `agent_state/bar.json`, total/done при resume пересчитываются из tasks.json.

### Поведение, которое обязательно должно получиться (critical requirements)
1. Три модели НИКОГДА не находятся в памяти одновременно: каждый вызов — отдельная «сессия» с keep_alive=0;
   переключение ролей полностью автоматически, пользователем не управляется.
2. Ни один байт проекта не пишется вне `workflow/<project-id>/`; там же `agent_state/` со всем
   состоянием; у проекта свой git-репозиторий, коммит на каждую принятую задачу.
3. Контекст любого запроса к модели ≤ ~11K токенов промпта (сборщик блоков + усечение вывода тестов
   до 4000 симв + фрагменты кода ≤1500 симв/файл + progress tail ≤40 строк).
4. Задача принимается ТОЛЬКО если реальные проверки прошли (exit code 0) И tester вернул passed=true.
5. Битый/не-JSON ответ модели не должен ронять процесс никогда: recovery в extract_json,
   chat_json-ретраи, нормализация задач, fallback-модель, «ошибка кодера → раунд/failed»,
   «ошибка тестера → decisive = hard checks», «LLMError → graceful resume».
6. Повторный запуск той же команды продолжает проект с первой незавершённой задачи (resume),
   включая восстановление прогресс-бара.
7. В конце генерируется `audit_report.md` со сверкой результата с планом; пробелы порождают
   дополнительный цикл реплана.

### Что проверить перед сдачей
- `python -m py_compile coding_agent/*.py`;
- мок-смоук оркестратора (подменить llm.chat): happy-path 2/2; debug-цикл с настоящим py_compile
  (битый код → раунд отладки → принят); resume (недоделанный проект подхватывается, счётчик бара корректен);
- юнит-тесты extract_json на грязных ответах: markdown-обёртка, голый массив файлов, литеральные
  \n внутри content, обрезанный ответ;
- README.md: описание архитектуры, таблица ролей/моделей, установка (ollama list, pip install -r
  requirements.txt), запуск (примеры для PowerShell), resume/--name/--fresh, структура workflow/,
  настройка через config.py/env, раздел про прогресс-бар, частые проблемы (в т.ч. замена qwen3.5:9b
  на qwen2.5-coder:7b при нехватке VRAM).

Не меняй имена файлов, модулей и конфигурационных констант — они заданы выше явно.
Библиотеки кроме `requests` не использовать. Тип-аннотации в стиле `str | None` (Python 3.10+).
