import unittest
from pyterrier_rag.prompt import Concatenator
class TestPyterrier_rag(unittest.TestCase):

    def test_T5(self):
        from pyterrier_rag.readers import T5FiD
        model = T5FiD(model_name_or_path="t5-base", tokenizer_name_or_path="t5-base")
        self._test_fid(model)

    def test_BART(self):
        from pyterrier_rag.readers import BARTFiD
        model = BARTFiD(model_name_or_path="facebook/bart-base", tokenizer_name_or_path="facebook/bart-base")
        self._test_fid(model)

    def test_FlanT5(self):
        from pyterrier_rag.readers import Reader
        from pyterrier_rag.backend import Seq2SeqLMBackend
        model = Seq2SeqLMBackend(model_name_or_path='google/flan-t5-base', return_logits=True)
        reader = Reader(model)
        qcontext_transformer = Concatenator()
        self._test_fid(reader, qcontext=qcontext_transformer)

    def _test_fid(self, model, qcontext=None):
        data = [
                {
                    "qid": "0",
                    "query": "where is Beijing?",
                    "docno": "d1",
                    "text": "New York is the capital city of USA.",
                    "score": 0.001
                },
                {
                    "qid": "0", 
                    "query": "where is Beijing?", 
                    "docno": "d2", 
                    "text": "Beijing is the capital city of China.", 
                    "score": 0.98
                },
            ]

        # now check without titles
        if qcontext is None:
            result = model(data)
            self.assertIn("China", result[0]['qanswer'])

            # test with titles
            data[0]['title'] = "USA"
            data[1]['title'] = "China"
            result = model(data)
            self.assertIn("China", result[0]['qanswer'])
        else:
            aggregate = qcontext(data)
            self.assertIn("Beijing", aggregate[0]['qcontext'])
            result = model(aggregate)
            self.assertIn("Beijing", result[0]['prompt'])
            self.assertIn("China", result[0]['qanswer'])