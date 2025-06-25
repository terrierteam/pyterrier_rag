import torch
import pytest
import sys
import types
import numpy as np

# assume the subclass is in vllm_backend.py in the same directory
from pyterrier_rag.backend._vllm import VLLMBackend
from pyterrier_rag.backend import BackendOutput


@unittest.skipIf(not torch.cuda.is_available(), "cuda not available")
class TestVllmBackend(test_backend.BaseTestBackend, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = VLLMBackend('HuggingFaceTB/SmolLM-135M')
