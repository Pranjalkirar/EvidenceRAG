"""Tests for evidencerag.langchain_impl.require_langchain -- the
optional-dependency guard, tested by intercepting `__import__` so this
file behaves the same whether or not LangChain is actually installed
in the environment running the tests.
"""

from __future__ import annotations

import builtins

import pytest

from evidencerag.langchain_impl import LangChainNotInstalledError, require_langchain


def test_langchain_not_installed_error_is_an_import_error():
    assert issubclass(LangChainNotInstalledError, ImportError)


def test_require_langchain_raises_actionable_error_when_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("langchain_core", "langchain", "langchain_community"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(LangChainNotInstalledError) as exc_info:
        require_langchain()

    message = str(exc_info.value)
    assert "langchain-core" in message
    assert "langchain-community" in message
    assert "pip install -e .[langchain]" in message


def test_require_langchain_reports_only_the_missing_packages(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "langchain_community":
            raise ImportError("No module named 'langchain_community'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(LangChainNotInstalledError) as exc_info:
        require_langchain()

    message = str(exc_info.value)
    assert "langchain-community" in message
    assert "langchain-core" not in message


def test_require_langchain_is_a_noop_when_everything_is_importable():
    pytest.importorskip("langchain_core")
    pytest.importorskip("langchain")
    pytest.importorskip("langchain_community")
    require_langchain()  # must not raise
