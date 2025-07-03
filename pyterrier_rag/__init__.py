"""Top-level package for PyTerrier RAG."""

__version__ = "0.2.1"

from pyterrier_rag import _datasets
from pyterrier_rag import measures
from pyterrier_rag import model
from pyterrier_rag import readers
from pyterrier_rag.search_o1 import SearchO1
from pyterrier_rag.search_r1 import SearchR1
from pyterrier_rag.r1_searcher import R1Searcher
from pyterrier_rag._util import ReasoningExtractor

from pyterrier_rag.backend import (
    OpenAIBackend,
    HuggingFaceBackend,
    Seq2SeqLMBackend,
    VLLMBackend,
    default_backend,
)

__all__ = [
    "Iterative",
    "model",
    "readers",
    "measures",
    "_datasets",
    "SearchO1",
    "SearchR1",
    "R1Searcher",
    "ReasoningExtractor",
    "OpenAIBackend",
    "HuggingFaceBackend",
    "Seq2SeqLMBackend",
    "VLLMBackend",
    "default_backend",
]
