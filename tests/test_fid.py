import unittest
import pandas as pd

class TestPyterrier_rag(unittest.TestCase):

    def test_T5(self):
        from pyterrier_rag.readers import T5FiD
        model = T5FiD(model_name_or_path="t5-base", tokenizer_name_or_path="t5-base")
        self._test_fid(model)

    def _test_fid(self, model):
        result = model(
            [
                {
                    "qid": "0", 
                    "query": "where is Beijing?", 
                    "docno": "d1", 
                    "title": "China", 
                    "text": "New York is the capital city of USA.", 
                    "score": 0.001
                },
                {
                    "qid": "0", 
                    "query": 
                    "where is Beijing?", 
                    "docno": "d2", 
                    "title": "USA", 
                    "text": "Beijing is the capital city of China.", 
                    "score": 0.98
                },  
            ]
        )
        print(result)