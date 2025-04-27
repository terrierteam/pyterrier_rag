import pytest
import sys
import types
import numpy as np

# assume the subclass is in vllm_backend.py in the same directory
from pyterrier_rag.backend._vllm import get_logits_from_dict, VLLMBackend
from pyterrier_rag.backend import BackendOutput

class DummyTokenizer:
    def get_vocab(self):
        # identity mapping: id -> id
        return {0: 0, 1: 1, 2: 2}


def test_get_logits_identity_mapping():
    tokenizer = DummyTokenizer()
    dlist = [{0: 0.5, 2: 1.5}, {1: 2.0}]
    out = get_logits_from_dict(dlist, tokenizer)
    expected = np.array([[0.5, 0.0, 1.5], [0.0, 2.0, 0.0]])
    np.testing.assert_array_equal(out, expected)


def test_vllmbackend_import_error(monkeypatch):
    # simulate vllm not available
    monkeypatch.setattr("pyterrier_rag._optional.is_vllm_availible", lambda: False)
    with pytest.raises(ImportError):
        VLLMBackend("model")


def test_vllmbackend_generate(monkeypatch):
    # simulate vllm available
    monkeypatch.setattr("pyterrier_rag._optional.is_vllm_availible", lambda: True)
    # create dummy vllm module
    dummy_vllm = types.SimpleNamespace()

    class DummyParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyOutput:
        def __init__(self, text, logprobs):
            self.text = text
            self.logprobs = logprobs

    class DummyResponse:
        def __init__(self, outputs):
            self.outputs = outputs

    class DummyLLM:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs

        def generate(self, inp, params):
            responses = []
            for prompt in inp:
                # reverse text, constant logprobs dict
                logprobs = {"a": 1.0, "b": 2.0}
                outputs = [DummyOutput(prompt[::-1], logprobs)]
                responses.append(DummyResponse(outputs))
            return responses

    dummy_vllm.LLM = DummyLLM
    dummy_vllm.SamplingParams = DummyParams
    sys.modules["vllm"] = dummy_vllm

    # instantiate backend
    backend = VLLMBackend(
        model_name_or_path="dummy-model",
        model_args={"opt": "x"},
        verbose=True,
    )
    # should have created DummyLLM
    assert isinstance(backend.model, DummyLLM)
    
    # test generate
    prompts = ["hello", "world"]
    outputs = backend.generate(prompts, extra_arg=123)
    assert isinstance(outputs, list)
    assert len(outputs) == 2
    for prompt, out in zip(prompts, outputs):
        assert isinstance(out, BackendOutput)
        # text should be reversed
        assert out.text == prompt[::-1]
        # logits should match dict
        assert isinstance(out.logits, dict)
        assert out.logits == {"a": 1.0, "b": 2.0}
