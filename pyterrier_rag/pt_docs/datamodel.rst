RAG Datamodel
=============

PyTerrier-RAG uses an extended datamodel, building on the standard PyTerrier datamodel (Q, D, R), adapted for answer generation.

+------+----------------------------+----------------------------------------------+
+ Type | Required Columns           | Description                                  +
+======+============================+==============================================+
|   A  |  ``["qid", "qanswer"]``    | Generated answers                            |
+------+---------+------------------+----------------------------------------------+
|   G  | ``["qid", "gold_answer"]`` | Gold truth answers (an array)                |
+------+--------------------     ---+----------------------------------------------+

Different transformer classes make different tranasformations between these datatypes:

 - Retriever: Q $\rightarrow$ R
 - Reranker: R  $\rightarrow$ R
 - Reader: R \rightarrow A
