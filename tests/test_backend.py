import itertools
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

    def generate(self, inp, return_logprobs=False, max_new_tokens=None, num_responses=1) -> Iterable[BackendOutput]:
        outputs = []
        for prompt in inp:
            for _ in range(num_responses):
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

    def test_generate(self):
        for return_logprobs, max_new_tokens, message_input, num_responses in itertools.product(
            [False, True],
            [None, 1],
            [False, True],
            [1, 2],
        ):
            if max_new_tokens is not None and not return_logprobs:
                continue # we can't really test this case unless we have logprobs
            inp = ['Hello', 'World', 'Some input']
            if message_input:
                inp = [
                    [
                        {'role': 'system', 'content': 'You are an intelligent system.'},
                        {'role': 'user', 'content': message},
                    ] for message in inp
                ]
            with self.subTest(return_logprobs=return_logprobs, max_new_tokens=max_new_tokens, message_input=message_input, num_responses=num_responses):
                if return_logprobs and not self.backend.supports_logprobs or message_input and not self.backend.supports_message_input or num_responses > 1 and not self.backend.supports_num_responses:
                    with self.assertRaises(ValueError):
                        self.backend.generate(inp, return_logprobs=return_logprobs, max_new_tokens=max_new_tokens, num_responses=num_responses)
                else:
                    outputs = self.backend.generate(inp, return_logprobs=return_logprobs, max_new_tokens=max_new_tokens, num_responses=num_responses)
                    self.assertIsInstance(outputs, list)
                    self.assertEqual(len(outputs), len(inp) * num_responses)
                    for output in outputs:
                        self.assertIsInstance(output, BackendOutput)
                        self.assertIsInstance(output.text, str)
                        self.assertFalse(output.text.startswith('ERROR::'), f"output error: {output.text}")
                        if return_logprobs:
                            self.assertIsInstance(output.logprobs, list)
                            if max_new_tokens is not None:
                                self.assertEqual(len(output.logprobs), max_new_tokens)
                            self.assertIsInstance(output.logprobs[0], dict)
                            self.assertNotEqual(len(output.logprobs[0]), 0)
                            first_logprob_key, first_logprob_value = list(output.logprobs[0].items())[0]
                            self.assertIsInstance(first_logprob_key, str)
                            self.assertIsInstance(first_logprob_value, float)
