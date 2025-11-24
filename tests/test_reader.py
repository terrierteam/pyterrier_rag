import pytest
import torch
import pyterrier as pt
import pandas as pd
from typing import Iterable
from pyterrier_rag.backend import Backend, BackendOutput
from pyterrier_rag.readers._base import Reader


# A minimal subclass implementing `generate` for testing
class DummyBackend(Backend):
    supports_logprobs = True

    def __init__(self, **kwargs):
        super().__init__("dummy-model", **kwargs)

    def generate(self, inp, return_logprobs=False, max_new_tokens=None, num_responses=1) -> Iterable[BackendOutput]:
        outputs = []
        for prompt in inp:
            for _ in range(num_responses):
                logprobs = [{'a': 1, 'b': 2}]
                outputs.append(
                    BackendOutput(
                        text=f"resp:{prompt}",
                        logprobs=logprobs,
                    )
                )
        return outputs


def dummy_prompt_fn(docs, query, **kwargs):
    """Simple prompt function that generates prompts from query and documents."""
    # docs is an iterator of (index, row) tuples
    doc_texts = []
    for _, doc in docs:
        if 'text' in doc.index:
            doc_texts.append(str(doc['text']))

    context = "\n".join(doc_texts) if doc_texts else ""
    if context:
        return f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    else:
        return f"Question: {query}\n\nAnswer:"


class TestReader:
    @pt.testing.transformer_test_class
    def test_reader():
        x = Reader(backend=DummyBackend(), prompt=dummy_prompt_fn)
        x.transform(pd.DataFrame(columns=['qid', 'query']))
        return x
