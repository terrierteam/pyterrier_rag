from abc import ABC
from typing import Iterable, Union

import pyterrier as pt
import torch
import numpy as np
from transformers import GenerationConfig
from more_itertools import chunked
from dataclasses import dataclass


@dataclass
class ReaderOutput:
    text: str = None
    logits: np.array = None


class Reader(pt.Transformer, ABC):
    _model_name_or_path = None
    _support_logits = False
    _logit_type = None

    def __init__(
        self,
        *,
        input_field: str = "query",
        output_field: str = "qanswer",
        output_format: str = "text",
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
        self.output_format = output_format
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

    def text_generator(self):
        return TextReader(self)

    def logit_generator(self):
        return LogitReader(self)

    def generate(self, inp: Iterable[str]) -> Iterable[Union[ReaderOutput, str]]:
        raise NotImplementedError("Implement the generate method")

    def transform_iter(self, inp: Iterable[dict]) -> Iterable[dict]:
        return self.text_generator()(inp)


class TextReader:
    def __init__(self, reader : Reader):
        self.reader = reader

    def __call__(self, inp: Iterable[str]) -> Iterable[str]:
        queries = [i[self.input_field] for i in inp]
        out = []
        for chunk in chunked(queries, self.batch_size):
            out.extend(self.reader.generate(chunk))
        if not hasattr(out[0], "text"):
            if not out[0] is str:
                raise ValueError("Reader must return ReaderOutput or str, not {}".format(type(out[0])))
        for i, o in zip(inp, out):
            i[self.output_field] = o.text
        return inp


class LogitReader:
    def __init__(self, reader : Reader):
        if not reader._support_logits:
            raise ValueError("Reader does not support logits")
        self.reader = reader

    def __call__(self, inp: Iterable[str]) -> Iterable[str]:
        queries = [i[self.input_field] for i in inp]
        out = []
        for chunk in chunked(queries, self.batch_size):
            out.extend(self.reader.generate(chunk))
        if not hasattr(out[0], "logits"):
            raise ValueError("Reader must return ReaderOutput to use LogitReader, not {}".format(type(out[0])))
        if out[0].logits is None:
            raise ValueError("Reader must return logits to use LogitReader")
        for i, o in zip(inp, out):
            i[self.output_field] = o.logits
        return inp


__all__ = ["Reader", "ReaderOutput", "TextReader", "LogitReader"]
