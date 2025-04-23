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

 - Retriever: Q $\rightarrow$ R -- retrieves documents in response to a query
 - Reranker: R  $\rightarrow$ R -- reranks documents for a given query
 - Reader: R $\rightarrow$ A -- generates an answer given retrieved documents
