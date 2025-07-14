PyTerrier RAG
=======================================================

`pyterrier-rag <https://github.com/terrierteam/pyterrier_rag>`__  is an extension for `PyTerrier <https://github.com/terrier-org/pyterrier>`__ that makes
it easier to produce retrieval augmented generation pipelines. PyTerrier-RAG supports:

#. Easy access to common QA datasets
#. Pre-built indices for common corpora
#. Popular reader models, such as Fusion-in-Decoder, LLama
#. Evaluation measures

As well as access to all of the retrievers (sparse, learned sparse, and dense) and rerankers (from MonoT5 to RankGPT) accessible through the wider `PyTerrier ecosystem <https://pyterrier.readthedocs.io/en/latest/>`__.

.. toctree::
    :maxdepth: 1

    datamodel
    measures
    backends

Example Notebooks
---------------------------------

Try out the following example notebooks to get started with PyTerrier RAG:

- Sparse Retrieval on Natural Questions with FiD and FlanT5 readers: `sparse_retrieval_FiD_FlanT5.ipynb <https://github.com/terrierteam/pyterrier_rag/blob/main/examples/nq/sparse_retrieval_FiD_FlanT5.ipynb>`_  
- Sparse Retrieval on Natural Questions with Mistral: `sparse_retrieval_Mistral.ipynb <https://github.com/terrierteam/pyterrier_rag/blob/main/examples/nq/sparse_retrieval_Mistral.ipynb>`_
- E5 Dense Retrieval with FiD on Natural Questions: `dense_e5_retrieval_FiD.ipynb <https://github.com/terrierteam/pyterrier_rag/blob/main/examples/nq/dense_e5_retrieval_FiD.ipynb>`_
- Agentic RAG: R1-Searcher `r1searcher.ipynb <https://github.com/terrierteam/pyterrier_rag/blob/main/examples/r1searcher.ipynb>`_
- Agentic RAG: Search-R1 `search-r1.ipynb <https://github.com/terrierteam/pyterrier_rag/blob/main/examples/search-r1.ipynb>`_
- Agentic RAG: Search-O1 `search-o1.ipynb <https://github.com/terrierteam/pyterrier_rag/blob/main/examples/search-o1.ipynb>`_

Credits
---------------------------------

- Craig Macdonald, University of Glasgow
- Jinyuan Fang, University of Glasgow
- Andrew Parry, University of Glasgow
- Zaiqiao Meng, University of Glasgow
- Sean MacAvaney, University of Glasgow
