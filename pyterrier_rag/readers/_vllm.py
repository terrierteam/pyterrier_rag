from typing import Iterable
import pandas as pd

from ._base import Reader
from .._optional import is_vllm_availible


class VLLMReader(Reader):
    def __init__(
        self,
        model_name_or_path: str,
        model_args: dict = {},
        generation_args: dict = None,
        batch_size: int = 4,
        max_input_length: int = 512,
        max_new_tokens: int = 32,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(
            batch_size=batch_size,
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
            generation_config=None,
            verbose=verbose,
            **kwargs,
        )
        if not is_vllm_availible():
            raise ImportError("Please install vllm to use VLLMReader")
        from vllm import LLM, EngineArgs, LLMEngine, SamplingParams

        self._model_name_or_path = model_name_or_path
        self._args = EngineArgs(model=model_name_or_path, **model_args)
        self._model = LLMEngine.from_engine_args(self._args)

        if isinstance(self._prompt, str):
            self._prompt = self._prompt.format

        if generation_args is None:
            generation_args = {
                "max_new_tokens": self.max_new_tokens,
                "temperature": 1.0,
                "do_sample": False,
                "num_beams": 1,
            }
        self._generation_args = SamplingParams(**generation_args)
        self.model = LLM(self._model, self._generation_args)

    def generate(self, inps: Iterable[str]) -> Iterable[str]:
        return map(
            lambda x: x[0].text, self.model.generate(inps, self._generation_args)
        )
