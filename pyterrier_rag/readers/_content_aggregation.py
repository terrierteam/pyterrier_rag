import pandas as pd
from typing import List, Optional, Any, Union, Tuple
import pyterrier as pt
import pyterrier_alpha as pta


def concat(
    input_texts: List[Union[str, Tuple]],
    intermediate_format: Optional[callable] = None,
    tokenizer: Any = None,
    max_length: int = -1,
    max_elements: int = -1,
    max_per_context: int = 512,
    truncation_rate: int = 50,
) -> str:
    if isinstance(input_texts[0], tuple):
        input_texts = ["\n".join(list(t)) for t in input_texts]
    if max_elements > 0:
        input_texts = input_texts[:max_elements]
    if tokenizer is not None:
        while True:
            total_context = ""
            for c in input_texts:
                if intermediate_format is not None:
                    if isinstance(c, tuple):
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
            total_context = "\n".join(map(intermediate_format, input_texts))
        else:
            total_context = "\n".join(input_texts)
    return total_context


def dataframe_concat(
    input_frame: pd.DataFrame = None,
    tokenizer: Any = None,
    max_length: int = -1,
    max_elements: int = -1,
    max_per_context: int = 512,
    truncation_rate: int = 50,
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


class ContentAggregationTransformer(pt.Transformer):
    def __init__(self,
                 aggregate_func: callable = concat,
                 in_fields: List[str] = ['text'],
                 out_field: str = "context",
                 per_query: bool = False):
        self.aggregate_func = aggregate_func
        self.in_fields = in_fields
        self.out_field = out_field
        self.per_query = per_query

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        inp[self.out_field] = self.aggregate_func(inp[self.in_fields].values)
        return inp


__all__ = ["concat", "dataframe_concat", "ContentAggregationTransformer"]
