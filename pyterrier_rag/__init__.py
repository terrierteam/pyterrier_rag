"""Top-level package for PyTerrier RAG."""

__version__ = "0.1.0"

import pyterrier_rag._datasets
import pyterrier_rag.measures
import pyterrier_rag.model
import pyterrier_rag.readers
import pyterrier_rag.llm
import pyterrier_rag.prompt

from pyterrier_rag.llm import (
    OpenAIBackend,
    CausalLMBackend,
    Seq2SeqLMBackend,
    VLLMBackend,
)

__all__ = [
    "readers",
    "prompt",
    "OpenAIBackend",
    "CausalLMBackend",
    "Seq2SeqLMBackend",
    "VLLMBackend",
]
