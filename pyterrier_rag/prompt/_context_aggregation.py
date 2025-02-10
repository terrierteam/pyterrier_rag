from typing import List, Optional, Any

import pandas as pd
import pyterrier as pt

from pyterrier_rag._util import dataframe_concat


class ContextAggregationTransformer(pt.Transformer):
    def __init__(self,
                 in_fields: Optional[List[str]] = ['text'],
                 out_field: Optional[str] = "context",
                 intermediate_format: Optional[callable] = None,
                 tokenizer: Optional[Any] = None,
                 max_length: Optional[int] = -1,
                 max_elements: Optional[int] = -1,
                 max_per_context: Optional[int] = 512,
                 truncation_rate: Optional[int] = 50,
                 aggregate_func: Optional[callable] = None,
                 per_query: bool = False
                 ):
        super().__init__()
        self.in_fields = in_fields
        self.out_field = out_field
        self.intermediate_format = intermediate_format
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_elements = max_elements
        self.max_per_context = max_per_context
        self.truncation_rate = truncation_rate
        self.aggregate_func = aggregate_func
        self.per_query = per_query

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        if self.aggregate_func is not None:
            if self.per_query:
                out = []
                for _, group in inp.groupby("query_id"):
                    group[self.out_field] = self.aggregate_func(group[self.in_fields])
                    out.append(group)
                return pd.concat(out)
            inp[self.out_field] = self.aggregate_func(inp[self.in_fields])
            return inp
        else:
            inp[self.out_field] = dataframe_concat(
                input_frame=inp,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                max_elements=self.max_elements,
                max_per_context=self.max_per_context,
                truncation_rate=self.truncation_rate,
                intermediate_format=self.intermediate_format,
                relevant_fields=self.in_fields,
                by_query=True,
            )
        return inp


__all__ = ["ContextAggregationTransformer"]
