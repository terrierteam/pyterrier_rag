import pytest
import sys
import types
import numpy as np

# assume the subclass is in vllm_backend.py in the same directory
from pyterrier_rag.backend._vllm import VLLMBackend
from pyterrier_rag.backend import BackendOutput


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
