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

Example Notebooks
=================

Try it out here on Google Colab now!

- Sparse Retrieval with FiD and FlanT5 readers: `sparse_retrieval_FiD_FlanT5.ipynb <sparse_retrieval_FiD_FlanT5.ipynb>`_  
  |Colab Badge|  

  .. |Colab Badge| image:: https://colab.research.google.com/assets/colab-badge.svg
     :target: https://colab.research.google.com/github/terrierteam/pyterrier_rag/blob/main/examples/nq/sparse_retrieval_FiD_FlanT5.ipynb

Credits
=======

- Craig Macdonald, University of Glasgow
- Jinyuan Fang, University of Glasgow
- Andrew Parry, University of Glasgow
- Zaiqiao Meng, University of Glasgow
- Sean MacAvaney, University of Glasgow
