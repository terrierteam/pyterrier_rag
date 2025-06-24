import types
import unittest
import os
import sys
import time
import pytest
import numpy as np
from pytest_httpserver import HTTPServer
from pyterrier_rag.backend import BackendOutput
from pyterrier_rag.backend._openai import OpenAIBackend
from openai.types.chat import ChatCompletion
from openai.types import Completion
from . import test_backend


@unittest.skipIf(os.environ.get('TEST_OPENAI_KEY') is None, "TEST_OPENAI_KEY not set")
class TestOpenAIBackend(test_backend.BaseTestBackend, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = OpenAIBackend(
            model_name_or_path="gpt-4o-mini",
            api_key=os.environ['TEST_OPENAI_KEY'],
            api='completions',
        )


@unittest.skipIf(os.environ.get('TEST_OPENAI_KEY') is None, "TEST_OPENAI_KEY not set")
class TestOpenAIBackendChat(test_backend.BaseTestBackend, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = OpenAIBackend(
            model_name_or_path="gpt-4o-mini",
            api_key=os.environ['TEST_OPENAI_KEY'],
            api='chat/completions',
        )


@unittest.skipIf(os.environ.get('TEST_IDA_KEY') is None, "TEST_IDA_KEY not set")
class TestOpenAIBackendLlama(test_backend.BaseTestBackend, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = OpenAIBackend(
            model_name_or_path="llama-3-8b-instruct",
            api_key=os.environ['TEST_IDA_KEY'],
            base_url='http://api.llm.apps.os.dcs.gla.ac.uk/v1/',
            api='completions',
        )


@unittest.skipIf(os.environ.get('TEST_IDA_KEY') is None, "TEST_IDA_KEY not set")
class TestOpenAIBackendLlamaChat(test_backend.BaseTestBackend, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = OpenAIBackend(
            model_name_or_path="llama-3-8b-instruct",
            api_key=os.environ['TEST_IDA_KEY'],
            base_url='http://api.llm.apps.os.dcs.gla.ac.uk/v1/',
            api='chat/completions',
        )


class TestOpenAIBackendMock(test_backend.BaseTestBackend, unittest.TestCase):
    @classmethod
    def completions(self, **kwargs):
        if kwargs.get('logprobs'):
            return Completion.construct(**{
                "choices": [
                    {"text": "world", "logprobs": {"top_logprobs": [{"a": 1, "b": 2}]}},
                ] * len(kwargs['prompt'])
            })
        else:
            return Completion.construct(**{
                "choices": [
                    {"text": "world"},
                ] * len(kwargs['prompt'])
            })

    @classmethod
    def setUpClass(cls):
        cls.backend = OpenAIBackend(
            model_name_or_path="dummy",
            api_key="dummy",
            api='completions',
        )
        # Simulate the OpenAI backend
        cls.backend.client = types.SimpleNamespace()
        cls.backend.client.completions = types.SimpleNamespace()
        cls.backend.client.completions.create = cls.completions


class TestOpenAIBackendMockChat(test_backend.BaseTestBackend, unittest.TestCase):
    @classmethod
    def chat_completions(self, **kwargs):
        if kwargs.get('logprobs'):
            return ChatCompletion.construct(**{
                "choices": [
                    {"message": {"content": "world"}, "logprobs": {'content': [{"top_logprobs": [{'token': 'a', 'logprob': 1}, {'token': 'b', 'logprob': 2}]}]}},
                ]
            })
        else:
            return ChatCompletion.construct(**{
                "choices": [
                    {"message": {"content": "world"}},
                ]
            })

    @classmethod
    def setUpClass(cls):
        cls.backend = OpenAIBackend(
            model_name_or_path="dummy",
            api_key="dummy",
            api='chat/completions',
        )
        # Simulate the OpenAI backend
        cls.backend.client = types.SimpleNamespace()
        cls.backend.client.chat = types.SimpleNamespace()
        cls.backend.client.chat.completions = types.SimpleNamespace()
        cls.backend.client.chat.completions.create = cls.chat_completions


def test_import_error_when_openai_not_available(monkeypatch):
    import importlib
    sys.modules.pop("some_module", None)
    real_import = __import__
    def fake_import(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("No module named 'openai'")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr("builtins.__import__", fake_import)

    try:
        importlib.reload(sys.modules.get('pyterrier_rag.backend._openai', None) or importlib.import_module('pyterrier_rag.backend._openai'))
        with pytest.raises(ImportError):
            OpenAIBackend("gpt-4o-mini")
    finally:
        importlib.reload(sys.modules.get('pyterrier_rag.backend._openai', None) or importlib.import_module('pyterrier_rag.backend._openai'))


def test_api_key(monkeypatch, subtests):
    with subtests.test(env=False, arg=False):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError):
            OpenAIBackend("gpt-4o-mini")

    with subtests.test(env=False, arg=True):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        backend = OpenAIBackend("gpt-4o-mini", api_key="sk-test")
        assert backend.client.api_key == "sk-test"

    with subtests.test(env=True, arg=False):
        monkeypatch.setenv("OPENAI_API_KEY", 'sk-env')
        backend = OpenAIBackend("gpt-4o-mini")
        assert backend.client.api_key == "sk-env"

    with subtests.test(env=True, arg=True):
        monkeypatch.setenv("OPENAI_API_KEY", 'sk-env')
        backend = OpenAIBackend("gpt-4o-mini", api_key="sk-test")
        assert backend.client.api_key == "sk-test"


def test_init(monkeypatch):
    OpenAIBackend("gpt-4o-mini", api_key="k")


def test_generate_chat_completions(monkeypatch, httpserver):
    httpserver.expect_request("/v1/chat/completions").respond_with_json({
        "choices": [
            {"message": {"content": "world"}},
        ]
    })

    backend = OpenAIBackend("gpt-4o-mini", base_url=httpserver.url_for('/v1/'), api_key="k")
    outputs = backend.generate(['hello', 'some message'])
    assert outputs == [BackendOutput(text='world'), BackendOutput(text='world')]


def test_generate_completions(monkeypatch, httpserver):
    httpserver.expect_request("/v1/completions").respond_with_json({
        "choices": [
            {"text": "world"},
            {"text": "universe"},
        ]
    })

    backend = OpenAIBackend("gpt-4o-mini", base_url=httpserver.url_for('/v1/'), api_key="k", api='completions')
    outputs = backend.generate(['hello', 'some message'])
    assert outputs == [BackendOutput(text='world'), BackendOutput(text='universe')]

TEST_OPENAI_BASE_URL = os.environ.get('TEST_OPENAI_BASE_URL')
TEST_OPENAI_API_KEY = os.environ.get('TEST_OPENAI_API_KEY')

@pytest.mark.skipif(not TEST_OPENAI_BASE_URL or not TEST_OPENAI_API_KEY, reason="TEST_OPENAI_BASE_URL and TEST_OPENAI_API_KEY not provided")
def test_integration_basic():
    backend = OpenAIBackend("gpt-4o-mini", base_url=TEST_OPENAI_BASE_URL, api_key=TEST_OPENAI_API_KEY)
    result = backend.generate(['hello world', 'some text'])
    assert len(result) == 2
    assert len(result[0].text) > 0
    assert len(result[1].text) > 0
