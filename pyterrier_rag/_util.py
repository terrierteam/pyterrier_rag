from typing import Union, Iterable, Tuple, List, Optional, Any, Callable
import pandas as pd
import pyterrier_alpha as pta
import itertools


def push_columns(
    df: pd.DataFrame,
    *,
    keep_original: bool = False,
    inplace: bool = False,
    base_column: str = "query",
) -> pd.DataFrame:
    """
    Changes a dataframe such that the selected column becomes "<column>_0", and any
    "<column>_0" columns becomes "<column>_1" etc.

    Arguments:
        df: Dataframe with a "<column>" column
        keep_original: if True, the <column> column is also left unchanged. Useful for client code.
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
            rename_cols[prev_col] = next_col
            prev_col = next_col
        else:
            break
    # apply renaming in-place or on the copy
    df.rename(columns=rename_cols, inplace=True)
    if keep_original:
        df[base_column] = df[f"{base_column}_0"]
    return df


def push_columns_dict(
    inp: Union[Iterable[dict], dict],
    keep_original: bool = False,
    base_column: str = "query",
) -> Union[Iterable[dict], dict]:
    def per_element(i: dict):
        cols = i.keys()
        if base_column not in cols:
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
      - For dict inputs, unpack as positional args (keys order).
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
    """
    if max_elements > 0:
        input_texts = input_texts[:max_elements]

    def to_text(element: Union[str, Tuple, List, dict]) -> str:
        return intermediate_formatting(element, intermediate_format)

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
            max_per_context = max(0, max_per_context - truncation_rate)

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
        if tokenizer is not None:
            max_context = max_per_context
            while True:
                total_context = ""
                for c in inp.itertuples():
                    if intermediate_format is not None:
                        c_text = intermediate_format({field: getattr(c, field, None) for field in relevant_fields})
                    else:
                        c_text = " ".join([getattr(c, field, None) for field in relevant_fields])
                    tokens = tokenizer.encode(c_text)
                    if len(tokens) > max_context:
                        tokens = tokens[:max_context]
                        c_text = tokenizer.decode(tokens)
                    total_context += c_text + "\n"
                if max_length < 0 or len(tokenizer.encode(total_context)) <= max_length:
                    return total_context.strip("\n")
                max_context = max(0, max_context - truncation_rate)
        else:
            # no-tokenizer path
            lines = []
            if intermediate_format is None:
                for row in inp.itertuples():
                    if len(relevant_fields) == 1:
                        val = getattr(row, relevant_fields[0], "")
                        lines.append(str(val))
                    else:
                        parts = [str(getattr(row, f, "")) for f in relevant_fields]
                        lines.append(" ".join(parts))
            else:
                for row in inp.itertuples():
                    d = {f: getattr(row, f, None) for f in relevant_fields}
                    lines.append(intermediate_format(d))
            return "\n".join(lines)

    def per_query_concat(inp: pd.DataFrame) -> pd.DataFrame:
        out = pta.DataFrameBuilder(["qid", "query", "qcontext"])
        for qid, group in inp.groupby("qid"):
            out.extend({
                "qid": [qid],
                "query": [group.iloc[0].query],
                "qcontext": [_concat(group)],
            })
        return out.to_df()

    if by_query:
        return per_query_concat(input_frame)
    return _concat(input_frame)
