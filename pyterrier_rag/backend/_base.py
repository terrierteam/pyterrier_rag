from abc import ABC
from typing import Iterable, Union, List

import pyterrier as pt
import torch
import numpy as np
from transformers import GenerationConfig
from more_itertools import chunked
from dataclasses import dataclass


@dataclass
class BackendOutput:
    text: str = None
    logits: np.array = None
    prompt_length: int = None


class Backend(pt.Transformer, ABC):
    _model_name_or_path = None
    _support_logits = False
    _logit_type = None
    _api_type = None
    _remove_prompt = False

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
        return TextBackend(self)

    def logit_generator(self):
        return LogitBackend(self)

    def _raw_generate(self, tokenized_sequences: Iterable[dict]) -> List[str]:
        """
        Generate text from the tokenized input sequences.
        This method is a placeholder and should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def generate(self, inp: Iterable[str]) -> Iterable[Union[BackendOutput, str]]:
        raise NotImplementedError("Implement the generate method")

    def transform_iter(self, inp: Iterable[dict]) -> Iterable[dict]:
        return self.text_generator(self).transform_iter(inp)


class TextBackend:
    def __init__(self, Backend: Backend):
        self.Backend = Backend

    def transform_iter(self, inp: Iterable[str]) -> Iterable[str]:
        queries = [i[self.input_field] for i in inp]
        out = []
        for chunk in chunked(queries, self.batch_size):
            out.extend(self.Backend.generate(chunk))
        if not hasattr(out[0], "text"):
            if not out[0] is str:
                raise ValueError(
                    "Backend must return BackendOutput or str, not {}".format(
                        type(out[0])
                    )
                )
        for i, o in zip(inp, out):
            i[self.output_field] = o.text
        return inp


class LogitBackend:
    def __init__(self, Backend: Backend):
        if not Backend._support_logits:
            raise ValueError("Backend does not support logits")
        self.Backend = Backend

    def transform_iter(self, inp: Iterable[str]) -> Iterable[str]:
        queries = [i[self.input_field] for i in inp]
        out = []
        for chunk in chunked(queries, self.batch_size):
            out.extend(self.Backend.generate(chunk))
        if not hasattr(out[0], "logits"):
            raise ValueError(
                "Backend must return BackendOutput to use LogitBackend, not {}".format(
                    type(out[0])
                )
            )
        if out[0].logits is None:
            raise ValueError("Backend must return logits to use LogitBackend")
        for i, o in zip(inp, out):
            i[self.output_field] = o.logits
        return inp


__all__ = ["Backend", "BackendOutput", "TextBackend", "LogitBackend"]
