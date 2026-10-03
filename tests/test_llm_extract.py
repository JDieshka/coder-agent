"""Юнит-тесты парсера JSON из ответов моделей (грязные/битые форматы)."""
import pytest

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
