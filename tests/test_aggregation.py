import unittest
import pandas as pd

class TestPyterrier_rag(unittest.TestCase):

    def test_content_aggregation_notok(self):
        import pyterrier_rag.readers._content_aggregation as ca
        
        # basic
        self.assertEqual("a\nb", ca.concat(['a', 'b']))

        # with titles
        self.assertEqual("a\n1\nb\n2", ca.concat([('a', '1'), ('b', '2')]))