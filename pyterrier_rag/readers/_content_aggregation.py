from itertools import chain
from typing import List, Tuple, Union


def concat(context : Union[ List[str], List[Tuple[str]]],
           tokenizer : str = None,
           max_length : str = -1,
           max_per_context : str = 512,
           truncation_rate : int = 50) -> str:
    """If tokenizer is not None,
    add all context to total context
    tokenize,
    if it exceeds max_length,
        truncate each context by 50 terms
    """
    if tokenizer is not None:
        while True:
            total_context = ""
            for c in context:
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
        total_context = "\n".join(chain(*context))
    return total_context


__all__ = ['concat']
