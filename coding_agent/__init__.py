"""Кодер-агент: Planner(qwen2.5-coder) -> Coder(qwen3.5) -> Tester/Debugger(qwen3) с ротацией моделей."""
from .orchestrator import main

__all__ = ["main"]
