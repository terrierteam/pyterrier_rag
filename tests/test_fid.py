import unittest
import pandas as pd

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
        from pyterrier_rag.readers import Seq2SeqLMReader
        model = Seq2SeqLMReader(model_name_or_path='google/flan-t5-base', model=None)
        self._test_fid(model)

    def _test_fid(self, model):
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
        result = model(data)
        self.assertTrue("China" in result[0]['qanswer'])

        # test with titles
        data[0]['title'] = "USA"
        data[1]['title'] = "China"
        result = model(data)
        self.assertTrue("China" in result[0]['qanswer'])