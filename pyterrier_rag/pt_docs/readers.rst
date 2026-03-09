.. _pyterrier_rag.readers:

Readers
==========================

Generic Reader
--------------------------

``Reader`` in the refactored prompt setup accepts either:

1. A jinja template string (rendered with ``docs`` plus query columns like ``query``), or
2. A callable prompt with signature like ``prompt(docs, query, **kwargs)``.

``docs`` is an iterator over the per-query retrieved rows (``group.iterrows()``).
The prompt can return either a single string or chat messages.

Example:

.. code-block:: python

    from pyterrier_rag.backend import Seq2SeqLMBackend
    from pyterrier_rag.readers import Reader

    reader = Reader(
        backend=Seq2SeqLMBackend("google/flan-t5-base"),
        prompt="Question: {{ query }}\nContext:{% for _, d in docs %}\n{{ d.text }}{% endfor %}\nAnswer:",
    )

.. autoclass:: pyterrier_rag.readers.Reader
   :members:
   :undoc-members:
   :show-inheritance:

Specific Readers
--------------------------

.. autoclass:: pyterrier_rag.readers.T5FiD
   :members:

.. autoclass:: pyterrier_rag.readers.BARTFiD
   :members:
