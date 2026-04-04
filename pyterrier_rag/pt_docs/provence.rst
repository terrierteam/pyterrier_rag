Provence Models
================

PyTerrier RAG includes Provence-style transformers for retrieval-augmented generation. These models
perform both re-ranking and pruning of retrieved passages in a single model call.

The current family includes:

- ``naver/provence-reranker-debertav3-v1``
- ``naver/xprovence-reranker-bgem3-v1``
- ``naver/xprovence-reranker-bgem3-v2``

For more information on Provence and, see the following papers:

.. cite.dblp:: conf/iclr/ChirkovaFNC25
.. cite.dblp:: journals/corr/abs-2601-18886

Basic usage
-----------

Use the Hugging Face model identifier directly when you want to be explicit about the checkpoint:

.. code-block:: python
    :caption: Create a Provence model directly from its Hugging Face checkpoint

    import pyterrier_rag as ptr

    provence = ptr.Provence("naver/provence-reranker-debertav3-v1")
    xprovence = ptr.Provence("naver/xprovence-reranker-bgem3-v2")

The transformer expects a result frame with at least ``qid``, ``query``, ``docno``, and ``text``. If a ``title``
column is present, it will be passed through to the underlying model as well. The output will be re-ranked (replacing
the ``score`` and ``rank`` columns) and the ``text`` of each result will be pruned to the portion of the passage that
is most relevant to the query.

.. code-block:: python
    :caption: Running Provence standalone

    provence(pd.DataFrame([
        {'qid': '1', 'query': "What is the capital of France?", 'docno': 'D1', 'text': "The capital of France is Paris.",},
        {'qid': '1', 'query': "What is the capital of France?", 'docno': 'D2', 'text': "Here is some content that doesn't matter and should be pruned from the results. The capital of France is Paris.",},
    ]))

    # qid                           query docno                             text     score  rank
    #   1  What is the capital of France?    D1  The capital of France is Paris.  6.383079     0
    #   1  What is the capital of France?    D2  The capital of France is Paris.  6.011631     1


API Documentation
-----------------

.. autoclass:: pyterrier_rag.Provence
   :members:
