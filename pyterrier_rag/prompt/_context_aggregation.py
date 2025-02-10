from typing import Iterable, List, Optional, Any, Union, Tuple

import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta


def concat(
    input_texts: List[Union[str, Tuple, List]],
    intermediate_format: Optional[callable] = None,
    tokenizer: Any = None,
    max_length: int = -1,
    max_elements: int = -1,
    max_per_context: int = 512,
    truncation_rate: int = 50,
) -> str:
    if max_elements > 0:
        input_texts = input_texts[:max_elements]
    if tokenizer is not None:
        while True:
            total_context = ""
            for c in input_texts:
                if intermediate_format is not None:
                    if not isinstance(c, str):
                        c = intermediate_format(*c)
                    else:
                        c = intermediate_format(c)
                tokens = tokenizer.encode(c)
                if len(tokens) > max_per_context:
                    tokens = tokens[:max_per_context]
                    text = tokenizer.decode(tokens)
                total_context += text + "\n"
            if len(tokenizer.encode(total_context)) <= max_length:
                break
            else:
                max_per_context -= truncation_rate
    else:
        if intermediate_format is not None:
            total_context = "\n".join(map(lambda x: intermediate_format(x) if isinstance(x, str) else intermediate_format(*x), input_texts))
        else:
            if not isinstance(input_texts[0], str):
                input_texts = ["\n".join(list(t)) for t in input_texts]
            total_context = "\n".join(input_texts)

    return total_context


def dataframe_concat(
    input_frame: pd.DataFrame = None,
    tokenizer: Any = None,
    max_length: int = -1,
    max_elements: int = -1,
    max_per_context: int = 512,
    truncation_rate: int = 25,
    intermediate_format: Optional[callable] = None,
    relevant_fields: Optional[list] = ["text"],
    by_query: bool = False,
) -> Union[str, pd.DataFrame, callable]:

    def _concat(inp: pd.DataFrame) -> str:
        if max_elements > 0:
            inp = inp.iloc[:max_elements]
        max_context = max_per_context
        if tokenizer is not None:
            while True:
                total_context = ""
                for c in inp.itertuples():
                    if intermediate_format is not None:
                        c = intermediate_format(
                            {
                                field: getattr(c, field, None)
                                for field in relevant_fields
                            }
                        )
                    else:
                        c = " ".join(
                            [getattr(c, field, None) for field in relevant_fields]
                        )
                    tokens = tokenizer.encode(c)
                    if len(tokens) > max_per_context:
                        tokens = tokens[:max_per_context]
                        text = tokenizer.decode(tokens)
                    total_context += text + "\n"
                if len(tokenizer.encode(total_context)) <= max_length:
                    break
                else:
                    max_context -= truncation_rate
        else:
            total_context = "\n".join(map(intermediate_format, inp))
        return total_context

    def per_query_concat(inp: pd.DataFrame) -> pd.DataFrame:
        out = pta.DataFrameBuilder(["query_id", "query", "context"])
        for qid, group in inp.groupby("query_id"):
            out.extend(
                {
                    "query_id": [qid],
                    "query": [group.iloc[0].query],
                    "context": [_concat(group)],
                }
            )
        return out.to_df()

    if by_query:
        return per_query_concat(input_frame)
    return _concat(input_frame)


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


__all__ = ["concat", "dataframe_concat", "ContextAggregationTransformer"]
