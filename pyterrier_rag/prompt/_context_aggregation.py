from typing import List, Optional, Any, Callable

import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta

from pyterrier_rag._util import concat


def score_sort(inp: List[dict]):
    if "score" in inp[0]:
        return sorted(inp, key=lambda x: x["score"], reverse=True)
    else:
        return inp


class Concatenator(pt.Transformer):
    """
    Transformer that concatenates specified fields from document records into a context string.

    At query time, orders, loads text (if needed), and aggregates records into a single context.

    Parameters:
        in_fields (List[str]): Fields to extract from each record. Defaults to ["text"].
        out_field (str): Name of the output context field. Defaults to "qcontext". 
        text_loader (Callable, optional): Function to load document text by doc ID.
        intermediate_format (Callable, optional): Formatter for individual records.
        tokenizer (Any, optional): Tokenizer used for length-based truncation.
        max_length (int): Max total token length of the context. Defaults to -1 (no limit).
        max_elements (int): Max number of records to include. Defaults to -1 (no limit).
        max_per_context (int): Max tokens per record.
        truncation_rate (int): Token drop rate during truncation. Defaults to 50.
        aggregate_func (Callable, optional): Custom aggregation function.
        ordering_func (Callable): Record ordering function before aggregation. 
            Defaults to score_sort, which sorts by "score" descending.

    Raises:
        ValueError: If 'text' is in in_fields but no text_loader is set.
    """
    def __init__(
        self,
        in_fields: Optional[List[str]] = ["text"],
        out_field: Optional[str] = "qcontext",
        text_loader: Optional[Callable] = None,
        intermediate_format: Optional[Callable] = None,
        tokenizer: Optional[Any] = None,
        max_length: Optional[int] = -1,
        max_elements: Optional[int] = -1,
        max_per_context: Optional[int] = -1,
        truncation_rate: Optional[int] = 50,
        aggregate_func: Optional[Callable] = None,
        ordering_func: Optional[Callable] = score_sort,
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

    def transform_inputs(self):
        if self.in_fields is None:
            return [['qid', 'query']]
        return [['qid', 'query'] + self.in_fields]

    def transform_outputs(self, inp_cols):
        return [self.out_field, 'qid', 'query']

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        pta.validate.columns(inp, includes=["qid"] + self.in_fields)
        output_frame = pta.DataFrameBuilder([self.out_field, "qid", "query"])

        if inp is None or inp.empty:
            return output_frame.to_df()

        for qid, group in inp.groupby("qid"):
            inp = group.to_dict(orient="records")
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
            output_frame.extend({self.out_field: context, "qid": qid, "query": query})
        return output_frame.to_df()


__all__ = ["Concatenator"]
