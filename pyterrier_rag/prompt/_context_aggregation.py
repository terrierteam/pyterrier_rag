from typing import List, Optional, Iterable

import pyterrier as pt
import pyterrier_alpha as pta

from ._config import ContextConfig
from pyterrier_rag._util import concat


class ContextAggregationTransformer(pt.Transformer):
    def __init__(
        self,
        config: Optional[ContextConfig] = None,
        in_fields: Optional[List[str]] = ["text"],
        out_field: Optional[str] = "context",
        aggregate_func: Optional[callable] = None,
    ):
        super().__init__()
        self.config = config
        assert (
            config is not None or aggregate_func is not None
        ), "Either a config or an aggregate function must be provided"
        if config is not None:
            self.in_fields = in_fields or config.in_fields
            self.out_field = out_field or config.out_field
            self.aggregate_func = aggregate_func or config.aggregate_func
            self.intermediate_format = config.intermediate_format
            self.tokenizer = config.tokenizer
            self.max_length = config.max_length
            self.max_elements = config.max_elements
            self.max_per_context = config.max_per_context
            self.truncation_rate = config.truncation_rate
        else:
            self.in_fields = in_fields
            self.out_field = out_field
            self.aggregate_func = aggregate_func

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
