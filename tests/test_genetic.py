import pyterrier as pt
from pyterrier_rag import Genetic
import pyterrier_dr
import unittest
import pandas as pd


class ConstMutator(pt.Transformer):
    def __init__(self, answer):
        self.answer = answer

    def transform(self, inp):
        return pd.DataFrame({'qid': [inp.qid[0]], 'qanswer': [self.answer]})


class TestGenetic(unittest.TestCase):
    def test_genetic(self):
        electra = pyterrier_dr.ElectraScorer(verbose=False)
        dataset = pt.get_dataset('irds:vaswani')
        index = pt.Artifact.from_hf('pyterrier/vaswani.terrier')
        mutators = [
            ConstMutator('Chemical reactions can happen sometimes'),
            ConstMutator('Chemical reactions happen when molecules collide with enough energy to break their existing bonds'),
        ]
        pipeline = index.bm25(num_results=5) >> dataset.text_loader() >> Genetic(electra, mutators)
        results = pipeline.search('when do chemical reactions happen')
        # the qanswer should be the good answer from above (with very high probability)
        self.assertEqual(results['qanswer'][0], 'Chemical reactions happen when molecules collide with enough energy to break their existing bonds')
