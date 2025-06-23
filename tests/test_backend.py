import pytest
import torch
import numpy as np
from typing import Iterable
from pyterrier_rag.backend import Backend, BackendOutput, TextBackend, LogitBackend
from transformers import GenerationConfig

# A minimal subclass implementing `generate` for testing
class DummyBackend(Backend):
    _model_name_or_path = "dummy-model"
    _support_logits = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.return_logits = True

    def generate(self, inp: Iterable[str]) -> Iterable[BackendOutput]:
        outputs = []
        for prompt in inp:
            logits = np.array([len(prompt), 0])
            outputs.append(
                BackendOutput(
                    text=f"resp:{prompt}",
                    logits=logits,
                    prompt_length=len(prompt),
                )
            )
        return outputs


def test_model_name_or_path_default():
    b = Backend()
    assert b.model_name_or_path is None


def test_model_name_or_path_subclass():
    b = DummyBackend()
    assert b.model_name_or_path == "dummy-model"


def test_default_device_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    b = DummyBackend()
    assert isinstance(b.device, torch.device)
    assert b.device == torch.device("cpu")


def test_default_device_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    b = DummyBackend()
    assert b.device == torch.device("cuda")


def test_device_arg_string(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    b = DummyBackend(device="cuda")
    assert b.device == torch.device("cuda")


def test_device_arg_device():
    b = DummyBackend(device=torch.device("cpu"))
    assert b.device == torch.device("cpu")


def test_generation_config_default(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    b = DummyBackend(max_new_tokens=42)
    cfg = b.generation_config
    assert cfg.max_new_tokens == 42
    assert cfg.temperature == 1.0
    assert cfg.do_sample is False
    assert cfg.num_beams == 1


def test_generation_config_custom(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    custom_cfg = GenerationConfig(
        max_new_tokens=5,
        temperature=0.5,
        do_sample=True,
        num_beams=3,
    )
    b = DummyBackend(generation_config=custom_cfg)
    assert b.generation_config is custom_cfg


def test_text_generator_returns_textbackend():
    b = DummyBackend()
    tb = b.text_generator()
    assert isinstance(tb, TextBackend)
    assert tb.backend is b


def test_logit_generator_without_support(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    class NoLogits(Backend):
        _support_logits = False
        def generate(self, inp):
            return [BackendOutput(text="x") for _ in inp]

    b = NoLogits()
    with pytest.raises(ValueError):
        b.logit_generator()


def test_logit_generator_with_support(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    b = DummyBackend()
    lb = b.logit_generator()
    assert isinstance(lb, LogitBackend)
    assert lb.backend is b


def test_backend_raw_generate_not_implemented(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    b = Backend()
    with pytest.raises(NotImplementedError):
        b._raw_generate([])


def test_backend_generate_not_implemented(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    b = Backend()
    with pytest.raises(NotImplementedError):
        b.generate([])


def test_textbackend_transform_iter_success(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    b = DummyBackend(batch_size=2)
    tb = TextBackend(b)
    inputs = [{"prompt": "one"}, {"prompt": "two"}, {"prompt": "three"}]
    result = tb.transform_iter(inputs)
    expected = [f"resp:{d['prompt']}" for d in inputs]
    for out_dict, exp in zip(result, expected):
        assert out_dict["qanswer"] == exp


def test_textbackend_transform_iter_invalid_type(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    class IntBackend(Backend):
        _support_logits = False
        def generate(self, inp):
            return [1 for _ in inp]

    tb = TextBackend(IntBackend())
    with pytest.raises(ValueError):
        tb.transform_iter([{"prompt": "x"}])


def test_logitbackend_transform_iter_success(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    b = DummyBackend(batch_size=2)
    lb = b.logit_generator()
    inputs = [{"prompt": "a"}, {"prompt": "bb"}]
    result = lb.transform_iter(inputs)
    for out_dict, inp in zip(result, inputs):
        np.testing.assert_array_equal(
            out_dict["qanswer"], np.array([len(inp["prompt"]), 0])
        )


def test_logitbackend_transform_iter_missing_logits_attr(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    class NoLogitAttr(Backend):
        _support_logits = True
        def generate(self, inp):
            class X: pass
            return [X() for _ in inp]
    with pytest.raises(ValueError):
        lb = NoLogitAttr().logit_generator()
        lb.transform_iter([{"prompt": "x"}])


def test_logitbackend_transform_iter_none_logits(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    class NoneLogits(Backend):
        _support_logits = True
        def generate(self, inp):
            return [BackendOutput(text="t", logits=None, prompt_length=1) for _ in inp]

    with pytest.raises(ValueError):
        lb = NoneLogits().logit_generator()
        lb.transform_iter([{"prompt": "x"}])
