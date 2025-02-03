from typing import Optional
from abc import abstractmethod

import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta

"""
Example Usage:

reader = VLLMReader("t5-base")
retriever = TasB
pipeline = retriever >> reader
exit_condition = lambda x : "the answer is" in x.iloc[0].qanswer.lower()
max_iter = 5
iterative = Iterative(pipeline, max_iter, exit_condition)
"""


class Iterative(pt.Transformer):
    def __init__(self, pipeline, max_iter: Optional[int] = None, exit_condition : Optional[callable] = lambda x : True,):
        self.pipeline = pipeline
        self.max_iter = max_iter
        self._exit_condition = exit_condition

    def _exceeded_max_iter(self, iter: int) -> bool:
        return self.max_iter is not None and iter == self.max_iter

    @pta.transform.by_query(add_ranks=False)
    def transform(self, inp : pd.DataFrame) -> pd.DataFrame:
        iter = 1
        stop = False
        while not stop:
            results = self.pipeline(inp)
            if self.exit_condition(results) or self._exceeded_max_iter(iter):
                stop = True
            inp = results
            iter += 1
        return results
