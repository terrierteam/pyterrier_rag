import unittest
import pandas as pd

class TestPyterrier_rag(unittest.TestCase):
    def test_something(self):
        import pyterrier as pt
        import pyterrier_rag
        #self.assertEqual(2, len(pt.get_dataset("rag:hotpotqa").get_topics('train').head(2)))