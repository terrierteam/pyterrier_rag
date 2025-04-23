RAG Measures
============

PyTerrier-RAG offers a number of commonly used evaluation measures:

For comparison with gold-truth answers:
 - Exact match percentage: ``pyterrier_rag.measures.EM``
 - F1: ``pyterrier_rag.measures.F1``
 - ROUGE: ``pyterrier_rag.measures.ROUGE1P``, ``pyterrier_rag.measures.ROUGE1R``, ``pyterrier_rag.measures.ROUGE1F`` etc, as implemented by the `rouge-score <https://pypi.org/project/rouge-score/>`_ library.


Example::

    pt.Experiment(
        [ragpipe1, ragpipe2],
        dataset.get_topics(),
        dataset.get_answers(),
        [pyterrier_rag.measures.EM, pyterrier_rag.measures.F1, pyterrier_rag.measures.ROUGE1F]
    )

Various ROUGE measures are available:
 - ROUGE-1 (precision, recall, f-measure)
 - ROUGE-2 (precision, recall, f-measure)
 - ROUGE-L (precision, recall, f-measure) 

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