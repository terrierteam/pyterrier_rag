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


def test_vllmbackend_generate():
    # instantiate backend
    backend = VLLMBackend(
        model_name_or_path="HuggingFaceTB/SmolLM-135M",
        verbose=True,
    )

    # test generate
    prompts = ["hello", "world"]
    outputs = backend.generate(prompts)
    assert isinstance(outputs, list)
    assert len(outputs) == 2
    for prompt, out in zip(prompts, outputs):
        assert isinstance(out, BackendOutput)
