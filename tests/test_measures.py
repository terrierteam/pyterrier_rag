import unittest
import pandas as pd
import pyterrier as pt
import pyterrier_rag

class TestPyterrier_rag(unittest.TestCase):

    def test_bertscore(self):
        run1 = pd.DataFrame([{'qid' : 'q1', 'qanswer' : 'chemical reactions happen when chemicals are mixed' }])
        run2 = pd.DataFrame([{'qid' : 'q1', 'qanswer' : 'chemical reactions were created by professor proton' }])
        qrels = pd.DataFrame([{'qid' : 'q1', 'docno' : 'd1', 'text' : 'chemical reactions occur when chemical are combined', 'label' : 2}])

        eval1 = pt.Evaluate(run1, qrels, [pyterrier_rag.measures.BERTScore(minlabel=2)])
        eval2 = pt.Evaluate(run2, qrels, [pyterrier_rag.measures.BERTScore(minlabel=2)])
        self.assertTrue(eval1['<lambda>'] > eval2['<lambda>'])