"""Юнит-тесты парсера JSON из ответов моделей (грязные/битые форматы)."""
import pytest

from coding_agent import llm
from coding_agent.llm import LLMError, extract_json


def test_plain_object():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_fenced_block():
    resp = 'Вот результат:\n```json\n{"passed": true, "diagnosis": ""}\n```\nготово'
    out = extract_json(resp)
    assert out["passed"] is True


def test_wrapped_files_object():
    resp = '{"files": [{"path": "app.py", "content": "print(1)"}], "notes": "ok"}'
    out = extract_json(resp)
    assert out["files"][0]["path"] == "app.py"


def test_bare_array_of_files():
    resp = '[{"path": "app.py", "content": "print(1)"}]'
    out = extract_json(resp)
    assert isinstance(out, list) and out[0]["path"] == "app.py"


def test_literal_newlines_fixed():
    resp = '{"files": [{"path": "a.py", "content": "x = 1\\ny = 2"}], "notes": "ok"}'
    out = extract_json(resp)
    assert out["files"][0]["content"] == "x = 1\ny = 2"


def test_truncated_array_repaired():
    # оборванный массив -> обёртка {"files":[полные элементы]}
    resp = '[{"path": "a.py", "content": "x"}, {"path": "b.py", "content": "y"'
    out = extract_json(resp)
    assert out["files"] == [{"path": "a.py", "content": "x"}]


def test_truncated_array_two_complete():
    resp = '[{"path": "a.py", "content": "x"}, {"path": "b.py", "content": "y"}, {"path": "c.py", "cont'
    out = extract_json(resp)
    assert len(out["files"]) == 2


def test_text_before_and_after_json():
    resp = 'Думаю... {"summary": "ok", "covered": [], "gaps": ["#2"]} ...надеемся'
    out = extract_json(resp)
    assert out["summary"] == "ok"


def test_invalid_raises_llmerror():
    with pytest.raises(LLMError):
        extract_json("совсем не json {{{")


def test_salvage_truncated_plan_tasks():
    """Обрезанный плановый JSON: полные задачи из оборванного массива tasks."""
    t = ('{"plan_md": "# План\\ntext", "tasks": '
         '[{"id":1,"title":"Каркас","goal":"g","files":["a.py"],"acceptance":"ok"},'
         '{"id":2,"title":"Auth","goal":"g2","files":["b.py"],"acc')
    r = llm.extract_json(t)
    assert isinstance(r, dict) and len(r["tasks"]) == 1
    assert r["tasks"][0]["title"] == "Каркас"


def test_truncated_plan_without_tasks_raises():
    """План оборван до начала tasks — предсказуемая ошибка (ретрай с коротким хинтом)."""
    t = '{"plan_md": "# План\\n## Стек\\n- Python\\nproject/\\n├── app/\\n│   ├── models.py'
    with pytest.raises(llm.LLMError):
        llm.extract_json(t)


def test_parse_plan_restores_plan_md_from_tasks():
    from coding_agent import roles
    t = ('{"plan_md": "# П\\nt", "tasks": [{"id":1,"title":"T1","goal":"g",'
         '"files":["x"],"acceptance":"a"},{"id":2,"title":"T2","goal":"g","files"')
    d = roles._parse_plan(t)
    assert d["plan_md"].startswith("# План") or "восстановлен" in d["plan_md"]
    assert len(d["tasks"]) == 1
