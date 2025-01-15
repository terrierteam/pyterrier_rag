from typing import Optional

import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta


class Iterative(pt.Transformer):

    def __init__(self, retriever : pt.Transformer, reader : pt.Transformer, max_iter : Optional[int] =None):
        self.retriever = retriever
        self.reader = reader
        self.max_iter = max_iter

    @pta.transform.by_query(add_ranks=False)
    def transform(self, inp : pd.DataFrame) -> pd.DataFrame:
        iter = 1
        stop = False
        while not stop:
            results = self.retriever(inp)
            answers = self.reader(results)
            # TODO is self.reader assumed to append LLM output to query?
            if self.max_iter is not None and iter == self.max_iter:
                stop = True
            # TODO should be more customisable - perhaps a lambda?
            if "the answer is" in answers.iloc[0].qanswer.lower():
                stop = True
            inp = answers
        return answers
