RAG Datamodel
=============

PyTerrier-RAG uses an extended datamodel, building on the standard PyTerrier datamodel (Q, D, R), adapted for answer generation.

+------+----------------------------+----------------------------------------------+
+ Type | Required Columns           | Description                                  +
+======+============================+==============================================+
|   A  |  ``["qid", "qanswer"]``    | Generated answers                            |
+------+---------+------------------+----------------------------------------------+
|  GA  | ``["qid", "gold_answer"]`` | Gold truth answers (an array)                |
+------+----------------------------+----------------------------------------------+

Different transformer classes make different tranasformations between these datatypes:

 - Retriever (Q :math:`\rightarrow` R) -- retrieves documents in response to a query. Example: ``pt.terrier.Retriever()``.
 - Reranker: (R :math:`\rightarrow` R) -- reranks retriever documents for a given query. Example: ``pyterrier_t5.MonoT5()``. 
 - 0-shot answer generation (Q :math:`\rightarrow` A) -- generates an answer without reference to any retrieved documents.  
 - Reader: R :math:`\rightarrow` A -- generates an answer given retrieved documents, ala RAG. Example ``pyterrier_rag.readers.T5FiD()``.
