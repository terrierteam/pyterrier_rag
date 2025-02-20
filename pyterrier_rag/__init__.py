"""Top-level package for PyTerrier RAG."""

__version__ = '0.1.0'

from _pyterrier_rag import _datasets
from pyterrier_rag import measures
from pyterrier_rag import model
from pyterrier_rag import readers
from pyterrier_rag._frameworks import Iterative

__all__ = [
    'Iterative', 'model', 'readers', 'measures', '_datasets',
]
