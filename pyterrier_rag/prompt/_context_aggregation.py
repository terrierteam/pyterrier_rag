from typing import List, Optional, Iterable, Any

import pyterrier as pt
import pyterrier_alpha as pta

from pyterrier_rag._util import concat


class ContextAggregationTransformer(pt.Transformer):
    def __init__(
        self,
        in_fields: Optional[List[str]] = ["text"],
        out_field: Optional[str] = "context",
        intermediate_format: Optional[callable] = None,
        tokenizer: Optional[Any] = None,
        max_length: Optional[int] = -1,
        max_elements: Optional[int] = -1,
        max_per_context: Optional[int] = -1,
        truncation_rate: Optional[int] = 50,
        aggregate_func: Optional[callable] = None,
    ):
        super().__init__()

        self.in_fields = in_fields
        self.out_field = out_field
        self.aggregate_func = aggregate_func
        self.intermediate_format = intermediate_format
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_elements = max_elements
        self.max_per_context = max_per_context
        self.truncation_rate = truncation_rate

    @pta.transform.by_query(add_ranks=False)
    def transform_iter(self, inp: Iterable[dict]) -> Iterable[dict]:
        return self.transform_by_query(inp)

    def transform_by_query(self, inp: Iterable[dict]) -> Iterable[dict]:
        qid = inp[0].get("qid", None)
        query = inp[0].get("query", None)

        relevant = [{k: v for k, v in i.items() if k in self.in_fields} for i in inp]
        if self.aggregate_func is not None:
            context = self.aggregate_func(relevant)
        else:
            context = concat(
                relevant,
                intermediate_format=self.intermediate_format,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                max_elements=self.max_elements,
                max_per_context=self.max_per_context,
                truncation_rate=self.truncation_rate,
            )
        return {self.out_field: context, "qid": qid, "query": query}


__all__ = ["ContextAggregationTransformer"]
