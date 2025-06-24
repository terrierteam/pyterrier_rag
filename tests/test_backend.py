import unittest
import pandas as pd
import pytest
import torch
import numpy as np
from typing import Iterable
from pyterrier_rag.backend import Backend, BackendOutput, TextGenerator
from transformers import GenerationConfig

# A minimal subclass implementing `generate` for testing
class DummyBackend(Backend):
    supports_logprobs = True

    def __init__(self, **kwargs):
        super().__init__("dummy-model", **kwargs)

    def generate(self, inp, return_logprobs=False, max_new_tokens=None) -> Iterable[BackendOutput]:
        outputs = []
        for prompt in inp:
            logprobs = [{'a': 1, 'b': 2}]
            outputs.append(
                BackendOutput(
                    text=f"resp:{prompt}",
                    logprobs=logprobs,
                )
            )
        return outputs


def test_text_generator_returns_textgenerator():
    b = DummyBackend()
    tb = b.text_generator()
    assert isinstance(tb, TextGenerator)
    assert tb.backend is b


def test_logprobs_generator_without_support(monkeypatch):
    class NoLogprobs(Backend):
        supports_logprobs = False
        def generate(self, inp):
            return [BackendOutput(text="x") for _ in inp]

    b = NoLogprobs("dummy-model")
    with pytest.raises(ValueError):
        b.logprobs_generator()


def test_logprobs_generator_with_support(monkeypatch):
    b = DummyBackend()
    lb = b.logprobs_generator()
    assert isinstance(lb, TextGenerator)
    assert lb.backend is b
    assert lb.logprobs_field == 'qanswer_logprobs'


def test_textgenerator_transform_iter_success(monkeypatch):
    b = DummyBackend()
    tb = TextGenerator(b)
    inputs = [{"prompt": "one"}, {"prompt": "two"}, {"prompt": "three"}]
    result = tb.transform_iter(inputs)
    expected = [f"resp:{d['prompt']}" for d in inputs]
    for out_dict, exp in zip(result, expected):
        assert out_dict["qanswer"] == exp


def test_logprobgenerator_transform_iter_success(monkeypatch):
    b = DummyBackend()
    lb = b.logprobs_generator()
    inputs = [{"prompt": "a"}, {"prompt": "bb"}]
    result = lb.transform_iter(inputs)
    for out_dict, inp in zip(result, inputs):
        assert out_dict['qanswer'].startswith('resp:')
        assert out_dict['qanswer_logprobs'] == [{'a': 1, 'b': 2}]


class BaseTestBackend:
    def test_generate(self):
        prompts = ["Hello", "World"]
        outputs = self.backend.generate(prompts)
        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), len(prompts))
        for output in outputs:
            self.assertIsInstance(output, BackendOutput)
            self.assertIsInstance(output.text, str)
            self.assertEqual(output.logprobs, None)

    def test_generate_logprobs(self):
        if not self.backend.supports_logprobs:
            self.skipTest(f"{self.backend!r} does not support logprobs")
        prompts = ["Hello", "World"]
        outputs = self.backend.generate(prompts, return_logprobs=True)
        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), len(prompts))
        for output in outputs:
            self.assertIsInstance(output, BackendOutput)
            self.assertIsInstance(output.text, str)
            self.assertIsInstance(output.logprobs, list)
            self.assertIsInstance(output.logprobs[0], dict)

    def test_generate_logprobs_1tok(self):
        if not self.backend.supports_logprobs:
            self.skipTest(f"{self.backend!r} does not support logprobs")
        prompts = ["Hello", "World"]
        outputs = self.backend.generate(prompts, return_logprobs=True, max_new_tokens=1)
        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), len(prompts))
        for output in outputs:
            self.assertIsInstance(output, BackendOutput)
            self.assertIsInstance(output.text, str)
            self.assertIsInstance(output.logprobs, list)
            self.assertEqual(len(output.logprobs), 1)
            self.assertIsInstance(output.logprobs[0], dict)
