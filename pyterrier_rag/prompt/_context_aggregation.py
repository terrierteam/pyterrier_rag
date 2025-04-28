from typing import List, Optional, Iterable, Any

import pyterrier as pt
import pyterrier_alpha as pta

from pyterrier_rag._util import concat


def score_sort(inp: List[dict]):
    if "score" in inp[0]:
        return sorted(inp, key=lambda x: x["score"], reverse=True)
    else:
        return inp


class ContextAggregationTransformer(pt.Transformer):
    def __init__(
        self,
        in_fields: Optional[List[str]] = ["text"],
        out_field: Optional[str] = "context",
        text_loader: Optional[callable] = None,
        intermediate_format: Optional[callable] = None,
        tokenizer: Optional[Any] = None,
        max_length: Optional[int] = -1,
        max_elements: Optional[int] = -1,
        max_per_context: Optional[int] = -1,
        truncation_rate: Optional[int] = 50,
        aggregate_func: Optional[callable] = None,
        ordering_func: Optional[callable] = score_sort,
    ):
        super().__init__()

        self.in_fields = in_fields
        self.out_field = out_field
        self.aggregate_func = aggregate_func
        self.text_loader = text_loader
        self.intermediate_format = intermediate_format
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_elements = max_elements
        self.max_per_context = max_per_context
        self.truncation_rate = truncation_rate
        self.ordering_func = ordering_func

    @pta.transform.by_query(add_ranks=False)
    def transform_iter(self, inp: Iterable[dict]) -> Iterable[dict]:
        return self.transform_by_query(inp)

    def transform_by_query(self, inp: Iterable[dict]) -> Iterable[dict]:
        inp = list(inp)
        qid = inp[0].get("qid", None)
        query = inp[0].get("query", None)
        if self.ordering_func is not None:
            inp = self.ordering_func(inp)
        relevant = [{k: v for k, v in i.items() if k in self.in_fields} for i in inp]
        if "text" in self.in_fields and "text" not in inp[0].keys():
            if self.text_loader is None:
                raise ValueError("Cannot retrieve text without a text loader")
            else:
                for d, t in zip(relevant, inp):
                    d["text"] = self.text_loader(t["docno"])

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
        return [{self.out_field: context, "qid": qid, "query": query}]


__all__ = ["ContextAggregationTransformer"]
