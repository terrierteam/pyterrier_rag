from abc import ABC
from typing import Iterable, Union

import pyterrier as pt
import torch
from transformers import GenerationConfig
from more_itertools import chunked


class Reader(pt.Transformer, ABC):
    _model_name_or_path = None

    def __init__(
        self,
        *,
        input_field: str = "query",
        output_field: str = "qanswer",
        batch_size: int = 4,
        max_input_length: int = 512,
        max_new_tokens: int = 32,
        generation_config: GenerationConfig = None,
        verbose: bool = False,
        device: Union[str, torch.device] = None,
        **kwargs
    ):
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.input_field = input_field
        self.output_field = output_field
        self.batch_size = batch_size
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose
        self.kwargs = kwargs

        if generation_config is None:
            # use greedy decoding by default
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_new_tokens,
                temperature=1.0,
                do_sample=False,
                num_beams=1,
            )
        else:
            self.generation_config = generation_config

    @property
    def model_name_or_path(self):
        return self._model_name_or_path

    def generate(self, inp: Iterable[str]) -> Iterable[str]:
        raise NotImplementedError("Implement the generate method")

    def transform_iter(self, inp: Iterable[dict]) -> Iterable[dict]:
        queries = [i[self.input_field] for i in inp]
        out = []
        for chunk in chunked(queries, self.batch_size):
            out.extend(self.generate(chunk))
        for i, o in zip(inp, out):
            i[self.output_field] = o
        return inp


__all__ = ["Reader"]
