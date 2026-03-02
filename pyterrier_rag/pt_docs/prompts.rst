Prompt Construction
==================  

This module provides classes for constructing prompts in a Retrieval-Augmented Generation (RAG) system. 
It includes functionality for aggregating context from multiple documents, and constructing prompts with
system messages etc. 

Legacy API
----------

``Concatenator``, ``PromptTransformer``, and ``prompt(...)`` are deprecated compatibility shims.
Prefer ``Reader(prompt=...)`` with ``jinja_formatter(...)`` and grouped-document prompt rendering.

.. autoclass:: pyterrier_rag.prompt.Concatenator
   :members:


.. autoclass:: pyterrier_rag.prompt.PromptTransformer
   :members:
