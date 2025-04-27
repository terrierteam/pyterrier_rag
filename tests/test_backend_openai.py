import os
import sys
import time
import pytest
from pyterrier_rag.backend import BackendOutput

import numpy as np

# assume the subclass is in openai_backend.py
from pyterrier_rag.backend._openai import OpenAIBackend


def test_import_error_when_openai_not_available(monkeypatch):
    # simulate openai not available
    monkeypatch.setattr("pyterrier_rag._optional.is_openai_availible", lambda: False)
    import importlib
    importlib.reload(sys.modules.get('pyterrier_rag.backend._openai', None) or importlib.import_module('pyterrier_rag.backend._openai'))
    with pytest.raises(ImportError):
        OpenAIBackend("gpt-4o-mini")


def test_value_error_when_no_api_key(monkeypatch):
    # simulate openai available but no API key
    monkeypatch.setattr("pyterrier_rag._optional.is_openai_availible", lambda: True)
    # ensure env var not set
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import importlib
    # reload openai_backend to reset import
    importlib.reload(sys.modules.get('pyterrier_rag.backend._openai', None) or importlib.import_module('pyterrier_rag.backend._openai'))
    with pytest.raises(ValueError):
        OpenAIBackend("gpt-4o-mini")


def test_api_key_from_parameter(monkeypatch):
    monkeypatch.setattr("pyterrier_rag._optional.is_openai_availible", lambda: True)
    # create dummy openai module
    class DummyChatCompletion:
        @staticmethod
        def create(*args, **kwargs):
            return {"choices": [{"message": {"content": ["ok"]}}]}

    dummy_openai = type(sys)("openai")
    dummy_openai.ChatCompletion = DummyChatCompletion
    sys.modules['openai'] = dummy_openai

    # ensure no env var
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = OpenAIBackend("gpt-4o-mini", api_key="sk-test")
    # openai.api_key should be set
    assert dummy_openai.api_key == "sk-test"
    assert backend._model_name_or_path == "gpt-4o-mini"
    # default generation_args set
    assert backend._generation_args["temperature"] == 1.0
    assert backend._generation_args["max_new_tokens"] == backend.max_new_tokens


def test_tokenizer_initialization(monkeypatch):
    monkeypatch.setattr("pyterrier_rag._optional.is_openai_availible", lambda: True)
    # simulate tiktoken available
    monkeypatch.setattr("pyterrier_rag._optional.is_tiktoken_availible", lambda: True)

    class DummyChatCompletion:
        @staticmethod
        def create(*args, **kwargs):
            return {"choices": [{"message": {"content": ["x"]}}]}

    dummy_openai = type(sys)("openai")
    dummy_openai.ChatCompletion = DummyChatCompletion
    sys.modules['openai'] = dummy_openai

    backend = OpenAIBackend("gpt-4o-mini", api_key="k")

def test_call_completion_success_after_retry(monkeypatch):
    monkeypatch.setattr("pyterrier_rag._optional.is_openai_availible", lambda: True)
    # dummy openai
    calls = []
    class DummyChat:
        @staticmethod
        def create(*args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise Exception("Temporary error")
            return {"choices": [{"message": {"content": "hello"}}]}
    dummy_openai = type(sys)("openai")
    dummy_openai.ChatCompletion = DummyChat
    sys.modules['openai'] = dummy_openai

    backend = OpenAIBackend("gpt-4o-mini", api_key="key")
    # patch sleep to avoid delay
    monkeypatch.setattr(time, "sleep", lambda s: None)
    # call private method
    result = backend._call_completion(messages=[{"role":"user","content":"hi"}], return_text=True)
    assert result == "hello"
    assert len(calls) == 2


def test_call_completion_reduce_length(monkeypatch):
    monkeypatch.setattr("pyterrier_rag._optional.is_openai_availible", lambda: True)
    class ChatErr:
        @staticmethod
        def create(*args, **kwargs):
            raise Exception("This model's maximum context length is")
    dummy_openai = type(sys)("openai")
    dummy_openai.ChatCompletion = ChatErr
    sys.modules['openai'] = dummy_openai

    backend = OpenAIBackend("gpt-4o-mini", api_key="k")
    res = backend._call_completion(messages=[], return_text=False)
    assert res == "ERROR::reduce_length"


def test_call_completion_filtered(monkeypatch):
    monkeypatch.setattr("pyterrier_rag._optional.is_openai_availible", lambda: True)
    class ChatErr2:
        @staticmethod
        def create(*args, **kwargs):
            raise Exception("The response was filtered")
    dummy_openai = type(sys)("openai")
    dummy_openai.ChatCompletion = ChatErr2
    sys.modules['openai'] = dummy_openai

    backend = OpenAIBackend("gpt-4o-mini", api_key="k")
    res = backend._call_completion(messages=[], return_text=False)
    assert res == "ERROR::The response was filtered"


def test_generate_outputs_backendoutput_list(monkeypatch):
    monkeypatch.setattr("pyterrier_rag._optional.is_openai_availible", lambda: True)
    # simulate openai returning list of strings
    class DummyChatCompletion3:
        @staticmethod
        def create(*args, **kwargs):
            return {"choices": [{"message": {"content": ["one","two"]}}]}
    dummy_openai = type(sys)("openai")
    dummy_openai.ChatCompletion = DummyChatCompletion3
    sys.modules['openai'] = dummy_openai

    backend = OpenAIBackend("gpt-4o-mini", api_key="k")
    prompts = [{"role":"user","content":"hi"}]
    outputs = backend.generate(prompts)
    assert isinstance(outputs, list)
    assert all(isinstance(o, BackendOutput) for o in outputs)
    texts = [o.text for o in outputs]
    assert texts == ["one", "two"]
