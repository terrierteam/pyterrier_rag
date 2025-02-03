from typing import Optional
from abc import abstractmethod

import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta


class IterativeTransformer(pt.Transformer):
    def __init__(self, max_iter: Optional[int] = None):
        self.max_iter = max_iter

    def _exceeded_max_iter(self, iter: int) -> bool:
        return self.max_iter is not None and iter == self.max_iter

    @abstractmethod
    def exit_condition(self, *args, **kwargs) -> bool:
        raise NotImplementedError("exit_condition must be implemented in a subclass")

    @abstractmethod
    def _inner_transform(self, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError("_inner_transform must be implemented in a subclass")

    @pta.transform.by_query(add_ranks=False)
    def transform(self, inp : pd.DataFrame) -> pd.DataFrame:
        iter = 1
        stop = False
        while not stop:
            results = self._inner_transform(inp)
            if self.exit_condition(results) or self._exceeded_max_iter(iter):
                stop = True
            inp = results
            iter += 1
        return results

class Iterative(IterativeTransformer):

    def __init__(self, retriever : pt.Transformer, reader : pt.Transformer, max_iter : Optional[int] =None):
        super().__init__(max_iter)
        self.retriever = retriever
        self.reader = reader

    def exit_condition(self, inp : pd.DataFrame) -> bool:
        return "the answer is" in inp.iloc[0].qanswer.lower()

    def _inner_transform(self, inp : pd.DataFrame) -> pd.DataFrame:
        results = self.retriever(inp)
        answers = self.reader(results)
        return answers
