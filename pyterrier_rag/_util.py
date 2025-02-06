from typing import Union, Iterable
import itertools


def push_queries_dict(inp: Union[Iterable[dict], dict], keep_originals: bool = False) -> Union[Iterable[dict], dict]:
    def per_element(i : dict):
        cols = i.keys()
        if "query" not in cols:
            raise KeyError(f"Expected a query column, but found {list(cols)}")
        prev_col = 'query'
        rename_cols = {}
        for query_idx in itertools.count():
            next_col = f'query_{query_idx}'
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
            renamed['query'] = renamed['query_0']

        return renamed

    if isinstance(inp, dict):
        return per_element(inp)
    return map(per_element, inp)
