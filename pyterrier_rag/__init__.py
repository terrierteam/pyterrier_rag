"""Top-level package for PyTerrier RAG."""

__version__ = '0.1.0'

import pyterrier_rag._datasets
import pyterrier_rag.measures
import pyterrier_rag.model
import pyterrier_rag.readers
from pyterrier_rag._frameworks import Iterative

__all__ = [
    'Iterative',
]
