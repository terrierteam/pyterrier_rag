import pandas as pd
from typing import List, Optional, Any, Union
import pyterrier_alpha as pta


def concat(input_texts: List[str],
           intermediate_format: Optional[callable] = None,
           tokenizer: Any = None,
           max_length: int = -1,
           max_per_context: int = 512,
           truncation_rate: int = 50) -> str:
    if tokenizer is not None:
        while True:
            total_context = ""
            for c in input_texts:
                if intermediate_format is not None:
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
        total_context = "\n".join(map(intermediate_format, input_texts))
    return total_context


def dataframe_concat(input_frame: pd.DataFrame = None,
                     tokenizer: Any = None,
                     max_length: int = -1,
                     max_per_context: int = 512,
                     truncation_rate: int = 50,
                     intermediate_format: Optional[callable] = None,
                     relevant_fields: Optional[list] = ['text'],
                     by_query: bool = False
                     ) -> Union[str, pd.DataFrame, callable]:

    def _concat(inp: pd.DataFrame) -> str:
        max_context = max_per_context
        if tokenizer is not None:
            while True:
                total_context = ""
                for c in inp.itertuples():
                    if intermediate_format is not None:
                        c = intermediate_format({field: getattr(c, field, None) for field in relevant_fields})
                    else:
                        c = ' '.join([getattr(c, field, None) for field in relevant_fields])
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
        out = pta.DataFrameBuilder(['query_id', 'query', 'context'])
        for qid, group in inp.groupby('query_id'):
            out.extend({
                'query_id': [qid],
                'query': [group.iloc[0].query],
                'context': [_concat(group)]
            })
        return out.to_df()

    if by_query:
        return per_query_concat(input_frame)
    return _concat(input_frame)


__all__ = ['concat', 'dataframe_concat']
