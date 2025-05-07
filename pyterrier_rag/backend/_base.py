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
    """
    Abstract base class for model-backed Transformers in PyTerrier.

    Subclasses must implement the raw generation logic (_raw_generate) and the
    high-level generate method. Supports optional logit extraction and prompt
    trimming.

    Parameters:
        input_field (str): Name of the input field carrying the prompt text.
        output_field (str): Name of the output field to populate with results.
        output_format (str): Desired format for text outputs (e.g., 'text').
        batch_size (int): Number of prompts to process per batch.
        max_input_length (int): Maximum token length for each input prompt.
        max_new_tokens (int): Maximum number of tokens to generate.
        generation_config (GenerationConfig): HuggingFace generation settings.
        verbose (bool): Flag to enable detailed logging.
        device (Union[str, torch.device]): Device for model execution.
    Attributes:
        _model_name_or_path: model name or checkpoint directory
        _support_logits (bool): Flag indicating logit support.
        _logit_type (str): Type of logits produced.
        _api_type (str): If using API do not return string
        _remove_prompt (bool): Whether to strip prompt tokens from decoded output.
    """
    
    _model_name_or_path = None
    _support_logits = False
    _logit_type = None
    _api_type = None
    _remove_prompt = False

    def __init__(
        self,
        *,
        input_field: str = "prompt",
        output_field: str = "qanswer",
        output_format: str = "text",
        batch_size: int = 4,
        max_input_length: int = 512,
        max_new_tokens: int = 32,
        return_logits: bool = False,
        generation_config: GenerationConfig = None,
        verbose: bool = False,
        device: Union[str, torch.device] = None,
        **kwargs,
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
        self.return_logits = return_logits
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
    
    @classmethod
    def from_model(self):
        raise NotImplementedError

    def text_generator(self):
        return TextBackend(self)

    def logit_generator(self):
        if not self.return_logits:
            raise ValueError("Cannot return logits as it is disabled")
        if not self._support_logits:
            raise ValueError("This model cannot return logits")
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


class TextBackend(pt.Transformer):
    def __init__(self, backend: Backend):
        self.backend = backend
        self.batch_size = self.backend.batch_size
        self.input_field = self.backend.input_field
        self.output_field = self.backend.output_field

    def transform_iter(self, inp: Iterable[str]) -> Iterable[str]:
        queries = [i[self.input_field] for i in inp]
        out = []
        for chunk in chunked(queries, self.batch_size):
            out.extend(self.backend.generate(chunk))
        if not hasattr(out[0], "text"):
            if out[0] is not str:
                raise ValueError("Backend must return BackendOutput or str, not {}".format(type(out[0])))
        for i, o in zip(inp, out):
            i[self.output_field] = o.text
        return inp


class LogitBackend(pt.Transformer):
    def __init__(self, backend: Backend):
        if not backend._support_logits:
            raise ValueError("Backend does not support logits")
        self.backend = backend
        self.batch_size = self.backend.batch_size
        self.input_field = self.backend.input_field
        self.output_field = self.backend.output_field

    def transform_iter(self, inp: Iterable[str]) -> Iterable[str]:
        queries = [i[self.input_field] for i in inp]
        out = []
        for chunk in chunked(queries, self.batch_size):
            out.extend(self.backend.generate(chunk))
        if not hasattr(out[0], "logits"):
            raise ValueError("Backend must return BackendOutput to use LogitBackend, not {}".format(type(out[0])))
        if out[0].logits is None:
            raise ValueError("Backend must return logits to use LogitBackend")
        for i, o in zip(inp, out):
            i[self.output_field] = o.logits
        return inp


__all__ = ["Backend", "BackendOutput", "TextBackend", "LogitBackend"]
