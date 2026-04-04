"""Top-level package for PyTerrier RAG."""

__version__ = "0.3.0"

from pyterrier_rag import _datasets
from pyterrier_rag import model
from pyterrier_rag.search_o1 import SearchO1
from pyterrier_rag.search_r1 import SearchR1
from pyterrier_rag.r1_searcher import R1Searcher
from pyterrier_rag._util import ReasoningExtractor
from pyterrier_rag.provence import Provence

try:
    from pyterrier_rag import measures
except ImportError:
    measures = None

try:
    from pyterrier_rag import readers
except ImportError:
    readers = None

try:
    from pyterrier_rag.frameworks import KnowledgeGraphExtractor, ReasoningChainGenerator
except ImportError:
    KnowledgeGraphExtractor = None
    ReasoningChainGenerator = None

from pyterrier_rag.backend import (
    Backend,
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
    "Provence",
    "Backend",
    "OpenAIBackend",
    "HuggingFaceBackend",
    "Seq2SeqLMBackend",
    "VLLMBackend",
    "default_backend",
    "KnowledgeGraphExtractor",
    "ReasoningChainGenerator",
]
