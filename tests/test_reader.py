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


class DummyPromptTransformer:
    def __init__(self, template):
        self.template = template
        self.input_fields = ['query']
        self.output_field = 'prompt'
        self.transform_calls = []

    def transform(self, df):
        self.transform_calls.append(df.copy())
        # Just return the prompt field for simplicity
        return pd.DataFrame({'prompt': df['query'].apply(lambda q: self.template.replace("{query}", q))})

    def __call__(self, df):
        return self.transform(df)


class TestReader:
    @pt.testing.transformer_test_class
    def test_reader():
        x = Reader(backend=DummyBackend(), prompt=DummyPromptTransformer("Answer the question: {query}"))
        x.transform(pd.DataFrame(columns=['qid', 'query']))
        return x
