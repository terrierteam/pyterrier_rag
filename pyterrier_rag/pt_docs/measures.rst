RAG Measures
============

PyTerrier-RAG offers a number of commonly used evaluation measures:

For comparison with gold-truth answers:
 - Exact match percentage: ``pyterrier_rag.measures.EM``
 - F1: ``pyterrier_rag.measures.F1``


Example::

    pt.Experiment(
        [ragpipe1, ragpipe2],
        dataset.get_topics(),
        dataset.get_answers(),
        [pyterrier_rag.measures.EM, pyterrier_rag.measures.F1]
    )

For comparison with known-relevant documents:
 - BERTScore (measures similarity of answer with relevant documents): ``pyterrier_rag.measures.BERTScore``

.. autofunction:: pyterrier_rag.measures.BERTScore()

Example::

    text_loader = pt.text.get_text(pt.get_dataset('irds:msmarco-passage'), 'text')
    topics_qrels = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')
    pt.Experiment(
        [ragpipe1, ragpipe2],
        dataset.get_topics('test-2019'),
        text_loader(dataset.get_qrels()),
        [pyterrier_rag.measures.BERTScore(rel=3)]
    )