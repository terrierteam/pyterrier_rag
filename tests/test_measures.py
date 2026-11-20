import unittest
from unittest.mock import patch
import pandas as pd
import pyterrier as pt
import pyterrier_rag

class TestPyterrier_rag(unittest.TestCase):

    def test_bertscore_gold_answers(self):
        run = pd.DataFrame([{'qid': 'q1', 'qanswer': 'model answer'}])
        answers = pd.DataFrame([
            {'qid': 'q1', 'gold_answer': 'first reference'},
            {'qid': 'q1', 'gold_answer': ['second reference', 'third reference']},
        ])
        with patch('pyterrier_rag._measures._bertscore_model') as scorer:
            scorer.compute.return_value = {
                'precision': [0.15, 0.3, 0.45],
                'recall': [0.3, 0.5, 0.7],
                'f1': [0.2, 0.4, 0.5],
            }
            measure = pyterrier_rag.measures.BERTScore(reference_source='answers', agg='avg')
            result = pt.Evaluate(run, answers, [measure])

        self.assertAlmostEqual(result['BERTScoreAnswers'], (0.2 + 0.4 + 0.5) / 3)
        self.assertEqual(scorer.compute.call_args.kwargs['predictions'], ['model answer'] * 3)
        self.assertEqual(scorer.compute.call_args.kwargs['references'],
                         ['first reference', 'second reference', 'third reference'])

    def test_bertscore_default_filters_document_judgments(self):
        run = pd.DataFrame([{'qid': 'q1', 'qanswer': 'model answer'}])
        qrels = pd.DataFrame([
            {'qid': 'q1', 'docno': 'd1', 'text': 'below threshold', 'label': 1},
            {'qid': 'q1', 'docno': 'd2', 'text': 'relevant document', 'label': 2},
        ])
        with patch('pyterrier_rag._measures._bertscore_model') as scorer:
            scorer.compute.return_value = {'precision': [0.4], 'recall': [0.8], 'f1': [0.5]}
            result = pt.Evaluate(run, qrels, [pyterrier_rag.measures.BERTScore(rel=2)])
        self.assertEqual(result['BERTScore'], 0.5)
        self.assertEqual(scorer.compute.call_args.kwargs['references'], ['relevant document'])

    def test_bertscore_gold_answers_selects_requested_submeasure(self):
        run = pd.DataFrame([{'qid': 'q1', 'qanswer': 'model answer'}])
        answers = pd.DataFrame([{'qid': 'q1', 'gold_answer': 'reference answer'}])
        with patch('pyterrier_rag._measures._bertscore_model') as scorer:
            scorer.compute.return_value = {'precision': [0.4], 'recall': [0.8], 'f1': [0.5]}
            measure = pyterrier_rag.measures.BERTScore(reference_source='answers', submeasure='recall')
            result = pt.Evaluate(run, answers, [measure])
        self.assertEqual(result['BERTScoreAnswers'], 0.8)

    def test_bertscore_can_evaluate_both_reference_sources(self):
        run = pd.DataFrame([{'qid': 'q1', 'qanswer': 'model answer'}])
        qrels = pd.DataFrame([
            {'qid': 'q1', 'docno': 'd1', 'label': 3, 'text': 'document', 'gold_answer': 'answer'},
        ])

        def compute(*, references, **kwargs):
            score = {'document': 0.3, 'answer': 0.7}[references[0]]
            return {component: [score] for component in ('precision', 'recall', 'f1')}

        with patch('pyterrier_rag._measures._bertscore_model') as scorer:
            scorer.compute.side_effect = compute
            measures = [pyterrier_rag.measures.BERTScore(),
                        pyterrier_rag.measures.BERTScore(reference_source='answers')]
            result = pt.Evaluate(run, qrels, measures)

        self.assertEqual(result, {'BERTScore': 0.3, 'BERTScoreAnswers': 0.7})

    def test_bertscore_gold_answers_rejects_missing_references(self):
        run = pd.DataFrame([{'qid': 'q1', 'qanswer': 'model answer'}])
        answers = pd.DataFrame([{'qid': 'q1', 'gold_answer': []}])
        measure = pyterrier_rag.measures.BERTScore(reference_source='answers')
        with patch('pyterrier_rag._measures._bertscore_model') as scorer:
            with self.assertRaisesRegex(ValueError, 'at least one text gold answer'):
                pt.Evaluate(run, answers, [measure])
            scorer.compute.assert_not_called()

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
