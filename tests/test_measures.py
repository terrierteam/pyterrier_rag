import unittest
import pandas as pd
import pyterrier as pt
import pyterrier_rag

class TestPyterrier_rag(unittest.TestCase):

    def test_rouge(self):
        run1 = pd.DataFrame([{'qid' : 'q1', 'qanswer' : 'chemical reactions happen when chemicals are mixed' }])
        run2 = pd.DataFrame([{'qid' : 'q1', 'qanswer' : 'chemical reactions were created by professor proton' }])
        qrels = pd.DataFrame([{'qid' : 'q1', 'gold_answer' : ['chemical reactions occur when chemical are combined']}])

        eval1 = pt.Evaluate(run1, qrels, [pyterrier_rag.measures.ROUGE1P])
        eval2 = pt.Evaluate(run2, qrels, [pyterrier_rag.measures.ROUGE1P])
        self.assertTrue(eval1['ROUGE1P'] > eval2['ROUGE1P'])
        
    def test_length(self):
        run = pd.DataFrame([{'qid' : 'q1', 'qanswer' : 'aaa' }, {'qid' : 'q2', 'qanswer' : '' }])
        qrels = pd.DataFrame([{'qid' : 'q1', 'docno' : 'd1', 'text' : 'chemical reactions occur when chemical are combined', 'label' : 2}])
        eval = pt.Evaluate(run, qrels, [pyterrier_rag.measures.AnswerLen, pyterrier_rag.measures.AnswerZeroLen ])
        self.assertEqual(1.5, eval["AnswerLen"])
        self.assertEqual(0.5, eval["AnswerZeroLen"])

    def test_bertscore(self):
        run1 = pd.DataFrame([{'qid' : 'q1', 'qanswer' : 'chemical reactions happen when chemicals are mixed' }])
        run2 = pd.DataFrame([{'qid' : 'q1', 'qanswer' : 'chemical reactions were created by professor proton' }])
        qrels = pd.DataFrame([{'qid' : 'q1', 'docno' : 'd1', 'text' : 'chemical reactions occur when chemical are combined', 'label' : 2}])

        eval1 = pt.Evaluate(run1, qrels, [pyterrier_rag.measures.BERTScore(rel=2)])
        eval2 = pt.Evaluate(run2, qrels, [pyterrier_rag.measures.BERTScore(rel=2)])
        self.assertTrue(eval1['BERTScore'] > eval2['BERTScore'])

    def test_bertscore_agg(self):
        run1 = pd.DataFrame([{'qid' : 'q1', 'qanswer' : 'chemical reactions happen when chemicals are mixed' }])
        qrels = pd.DataFrame([
            {'qid' : 'q1', 'docno' : 'd1', 'text' : 'chemical reactions occur when chemical are combined', 'label' : 2},
            {'qid' : 'q1', 'docno' : 'd2', 'text' : 'chemical reactions were created by professor proton', 'label' : 2},
            ])

        eval1 = pt.Evaluate(run1, qrels, [pyterrier_rag.measures.BERTScore(rel=2)])
