from abc import ABC, abstractmethod
from typing import Iterable, Union

import pyterrier as pt
import numpy as np
from more_itertools import chunked
from dataclasses import dataclass


@dataclass
class BackendOutput:
    text: str = None
    logits: np.array = None
    prompt_length: int = None


class Backend(pt.Transformer, ABC):
    """
    Abstract base class for model-backed Transformers in PyTerrier.

    Subclasses must implement the raw generation logic (generate) and the
    high-level generate method. Supports optional logit extraction and prompt
    trimming.

    Parameters:
        max_input_length (int): Maximum token length for each input prompt.
        max_new_tokens (int): Maximum number of tokens to generate.
        verbose (bool): Flag to enable detailed logging.
        device (Union[str, torch.device]): Device for model execution.
    Attributes:
        _model_name_or_path: model name or checkpoint directory
        _support_logits (bool): Flag indicating logit support.
        _logit_type (str): Type of logits produced.
        _api_type (str): If using API do not return string
    """

    _model_name_or_path = None
    _support_logits = False
    _logit_type = None
    _api_type = None

    def __init__(
        self,
        model_name_or_path: str,
        *,
        max_input_length: int = 512,
        max_new_tokens: int = 32,
        verbose: bool = False,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose

    # Abstract methods

    @abstractmethod
    def generate(self, inp: Iterable[str]) -> Iterable[Union[BackendOutput, str]]:
        raise NotImplementedError("Implement the generate method")

    # Transformer implementations

    def text_generator(self, *, input_field='prompt', output_field='qanswer') -> pt.Transformer:
        return TextGenerator(self, input_field=input_field, output_field=output_field)

    def logit_generator(self, *, input_field='prompt', output_field='qanswer') -> pt.Transformer:
        if not self._support_logits:
            raise ValueError("This model cannot return logits")
        return LogitGenerator(self, input_field=input_field, output_field=output_field)

    def transform_iter(self, inp: Iterable[dict]) -> Iterable[dict]:
        return self.text_generator().transform_iter(inp)


class TextGenerator(pt.Transformer):
    def __init__(self,
        backend: Backend,
        *,
        input_field: str = 'prompt',
        output_field: str = 'qanswer',
        batch_size: int = 4,
    ):
        self.backend = backend
        self.input_field = input_field
        self.output_field = output_field
        self.batch_size = batch_size

    def transform_iter(self, inp: Iterable[dict]) -> Iterable[dict]:
        for chunk in chunked(inp, self.batch_size):
            chunk = list(chunk)
            prompts = [i[self.input_field] for i in inp]
            out = self.backend.generate(prompts)
            for rec, o in zip(chunk, out):
                yield {**rec, self.output_field: o.text}


class LogitGenerator(pt.Transformer):
    def __init__(self,
        backend: Backend,
        *,
        input_field: str = 'prompt',
        output_field: str = 'qanswer',
        batch_size: int = 4,
    ):
        if not backend._support_logits:
            raise ValueError("Backend does not support logits")
        self.backend = backend
        self.input_field = input_field
        self.output_field = output_field
        self.batch_size = batch_size

    def transform_iter(self, inp: Iterable[dict]) -> Iterable[dict]:
        for chunk in chunked(inp, self.batch_size):
            chunk = list(chunk)
            prompts = [i[self.input_field] for i in inp]
            out = self.backend.generate(prompts, return_logits=True)
            for rec, o in zip(chunk, out):
                yield {**rec, self.output_field: o.logits}


__all__ = ["Backend", "BackendOutput", "TextGenerator", "LogitGenerator"]
