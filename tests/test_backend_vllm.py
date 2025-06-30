import unittest
import torch
import pytest
import sys
import types
import gc
import numpy as np

# assume the subclass is in vllm_backend.py in the same directory
from pyterrier_rag.backend._vllm import VLLMBackend
from pyterrier_rag.backend import BackendOutput, Backend
from . import test_backend


@unittest.skipIf(not torch.cuda.is_available(), "cuda not available")
class TestVllmBackend(test_backend.BaseTestBackend, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = Backend.from_dsn('vllm:HuggingFaceTB/SmolLM-135M')

    @classmethod
    def tearDownClass(cls):
        backend = cls.backend
        cls.backend = None
        del backend
        # Run garbage collection
        gc.collect()

        # Clear PyTorch CUDA cache
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
