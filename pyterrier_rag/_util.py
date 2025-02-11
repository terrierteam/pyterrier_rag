from typing import Union, Iterable, Tuple, List, Optional, Any
import pandas as pd
import itertools


def push_queries(df: pd.DataFrame,
                 *,
                 keep_original: bool = False,
                 inplace: bool = False,
                 base_column: str = 'query') -> pd.DataFrame:
    """
        Changes a dataframe such that the "query" column becomes "query_0", and any
        "query_0" columns becames "query_1" etc.

        Arguments:
            df: Dataframe with a "query" column
            keep_original: if True, the query column is also left unchanged. Useful for client code. 
                Defaults to False.
            inplace: if False, a copy of the dataframe is returned. If True, changes are made to the
                supplied dataframe. Defaults to False. 
    """
    cols = set(df.columns)
    if "query" not in cols:
        raise KeyError(f"Expected a {base_column} column, but found {list(cols)}")
    if not inplace:
        df = df.copy()
    prev_col = base_column
    rename_cols = {}
    for query_idx in itertools.count():
        next_col = f'{base_column}_{query_idx}'
        if prev_col in cols:
            rename_cols[prev_col] = next_col # map e.g., query_0 to be renamed to query_1
            prev_col = next_col
        else:
            break
    df = df.rename(columns=rename_cols)
    if keep_original:
        df[base_column] = df[f"{base_column}_0"]
    return df


def push_queries_dict(inp: Union[Iterable[dict], dict],
                      keep_originals: bool = False,
                      base_column: str = 'query') -> Union[Iterable[dict], dict]:
    def per_element(i: dict):
        cols = i.keys()
        if "query" not in cols:
            raise KeyError(f"Expected a {base_column} column, but found {list(cols)}")
        prev_col = base_column
        rename_cols = {}
        for query_idx in itertools.count():
            next_col = f'{base_column}_{query_idx}'
            if prev_col in cols:
                rename_cols[prev_col] = next_col
                prev_col = next_col
            else:
                break

        renamed = {}
        for k, v in i.items():
            if k in rename_cols:
                renamed[rename_cols[k]] = v
            elif keep_originals:
                renamed[k] = v

        if keep_originals:
            renamed[base_column] = renamed[f'{base_column}_0']

        return renamed

    if isinstance(inp, dict):
        return per_element(inp)
    return map(per_element, inp)


def find_maximum_push(inp: pd.DataFrame,
                      base_column: str = 'query') -> Tuple[str, int]:
    columns = inp.columns
    maxcol = None
    maxval = -1
    for col in columns:
        if col.startswith(f"{base_column}_"):
            val = int(col.split("_")[1])
            if val > maxval:
                maxval = val
                maxcol = col
    return maxcol, maxval


def find_maximum_push_dict(inp: Union[Iterable[dict], dict],
                           base_column: str = 'query') -> Tuple[str, int]:
    def per_element(i: dict):
        cols = i.keys()
        maxcol = None
        maxval = -1
        for col in cols:
            if col.startswith(f"{base_column}_"):
                val = int(col.split("_")[1])
                if val > maxval:
                    maxval = val
                    maxcol = col
        return maxcol, maxval

    if isinstance(inp, dict):
        return per_element(inp)
    return map(per_element, inp)


def intermediate_formatting(inp: Union[str, Tuple, List, dict],
                            intermediate_format: Optional[callable] = None) -> str:
    if intermediate_format is not None:
        if isinstance(inp, dict):
            return intermediate_format(**inp)
        elif isinstance(inp, list) or isinstance(inp, tuple):
            return intermediate_format(*inp)
        else:
            return intermediate_format(inp)
    return inp


def concat(
    input_texts: List[Union[str, Tuple, List, dict]],
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
                c = intermediate_formatting(c, intermediate_format)
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
            total_context = "\n".join(map(lambda x: intermediate_formatting(x, intermediate_format), input_texts))
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
