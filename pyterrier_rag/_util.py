from typing import Union, Iterable, Tuple, List, Optional, Any, Callable
import pandas as pd
import pyterrier_alpha as pta
import itertools


def push_queries(
    df: pd.DataFrame,
    *,
    keep_original: bool = False,
    inplace: bool = False,
    base_column: str = "query",
) -> pd.DataFrame:
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
    if base_column not in cols:
        raise KeyError(f"Expected a {base_column} column, but found {list(cols)}")
    if not inplace:
        df = df.copy()
    prev_col = base_column
    rename_cols = {}
    for query_idx in itertools.count():
        next_col = f"{base_column}_{query_idx}"
        if prev_col in cols:
            rename_cols[prev_col] = next_col  # map e.g., query_0 to be renamed to query_1
            prev_col = next_col
        else:
            break
    df = df.rename(columns=rename_cols)
    if keep_original:
        df[base_column] = df[f"{base_column}_0"]
    return df


def push_queries_dict(
    inp: Union[Iterable[dict], dict],
    keep_original: bool = False,
    base_column: str = "query",
) -> Union[Iterable[dict], dict]:
    def per_element(i: dict):
        cols = i.keys()
        if "query" not in cols:
            raise KeyError(f"Expected a {base_column} column, but found {list(cols)}")
        prev_col = base_column
        rename_cols = {}
        for query_idx in itertools.count():
            next_col = f"{base_column}_{query_idx}"
            if prev_col in cols:
                rename_cols[prev_col] = next_col
                prev_col = next_col
            else:
                break

        renamed = {}
        for k, v in i.items():
            if k in rename_cols:
                renamed[rename_cols[k]] = v
            else:
                renamed[k] = v

        if keep_original:
            renamed[base_column] = renamed[f"{base_column}_0"]

        return renamed

    if isinstance(inp, dict):
        return per_element(inp)
    return [*map(per_element, inp)]


def find_maximum_push(inp: pd.DataFrame, base_column: str = "query") -> Tuple[str, int]:
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


def find_maximum_push_dict(inp: Union[Iterable[dict], dict], base_column: str = "query") -> Tuple[str, int]:
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


def intermediate_formatting(
    inp: Union[str, Tuple, List, dict], intermediate_format: Optional[Callable[..., str]] = None
) -> str:
    """
    If an intermediate_format is provided, apply it:
      - For dict inputs, unpack as keyword args.
      - For tuple/list inputs, unpack as positional args.
      - For other inputs, pass directly.
    If no intermediate_format, and inp is a dict, return its 'text' field or empty string.
    Otherwise return inp unmodified (as string).
    """
    if intermediate_format is not None:
        if isinstance(inp, dict):
            return intermediate_format(**inp)
        if isinstance(inp, (list, tuple)):
            return intermediate_format(*inp)
        return intermediate_format(inp)

    # Default behavior when no formatter:
    if isinstance(inp, dict):
        return str(inp.get("text", ""))
    return str(inp)


def concat(
    input_texts: List[Union[str, Tuple, List, dict]],
    intermediate_format: Optional[Callable[..., str]] = None,
    tokenizer: Any = None,
    max_length: int = -1,
    max_elements: int = -1,
    max_per_context: int = 512,
    truncation_rate: int = 50,
) -> str:
    """
    Concatenate input_texts into a single string, applying optional formatting and token-based truncation.

    - If intermediate_format is None, dicts default to their 'text' field.
    - Tokenizer-based path enforces per-context and overall token limits via truncation.
    """
    if max_elements > 0:
        input_texts = input_texts[:max_elements]

    def to_text(element: Union[str, Tuple, List, dict]) -> str:
        return intermediate_formatting(element, intermediate_format)

    # When a tokenizer is provided, enforce token limits
    if tokenizer is not None:
        while True:
            segments: List[str] = []
            for elem in input_texts:
                text = to_text(elem)
                tokens = tokenizer.encode(text)
                if max_per_context > 0 and len(tokens) > max_per_context:
                    tokens = tokens[:max_per_context]
                    text = tokenizer.decode(tokens)
                segments.append(text)
            combined = "\n".join(segments)
            if max_length < 0 or len(tokenizer.encode(combined)) <= max_length:
                return combined
            # reduce per-context limit and retry
            max_per_context = max(0, max_per_context - truncation_rate)

    # No tokenizer: simple join
    texts = [to_text(x) for x in input_texts]
    return "\n".join(texts)


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
                        c = intermediate_format({field: getattr(c, field, None) for field in relevant_fields})
                    else:
                        c = " ".join([getattr(c, field, None) for field in relevant_fields])
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
