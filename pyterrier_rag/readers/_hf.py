from typing import Any, List

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
from . import _content_aggregation as content_aggregation
from ._base import GENERIC_PROMPT, Reader


class HuggingFaceReader(Reader):
    _prompt = GENERIC_PROMPT
    _model_class = None

    def __init__(
        self,
        model_name_or_path: str,
        model_args: dict = {},
        generation_args: dict = None,
        context_aggregation: str = "concat",
        prompt: Any = None,
        batch_size: int = 4,
        text_field: str = "text",
        max_input_length: int = None,
        num_context: int = 5,
        max_new_tokens: int = 32,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(
            batch_size=batch_size,
            text_field=text_field,
            text_max_length=max_input_length,
            num_context=num_context,
            max_new_tokens=max_new_tokens,
            generation_config=None,
            verbose=verbose,
            **kwargs,
        )
        self._model_name_or_path = model_name_or_path
        self._model = (
            None
            if self._model_class is None
            else self._model_class.from_pretrained(model_name_or_path, **model_args)
            .to(self.device)
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.max_input_length = (
            max_input_length or self._model.config.max_position_embeddings
        )

        if context_aggregation not in content_aggregation.__all__:
            raise ValueError(
                f"context_aggregation must be one of {content_aggregation.__all__}"
            )
        self._context_aggregation = getattr(content_aggregation, context_aggregation)
        self._prompt = prompt or self._prompt

        if isinstance(self._prompt, str):
            self._prompt = self._prompt.format

        if generation_args is None:
            generation_args = {
                "max_new_tokens": self.max_new_tokens,
                "temperature": 1.0,
                "do_sample": False,
                "num_beams": 1,
            }
        self._generation_args = generation_args
        self.model = self._model

    def generate(self, inps: List[str]) -> List[str]:
        assert (
            self.model is not None
        ), "Model is not loaded, you should instantiate a subclass of HFModel"
        inputs = self.tokenizer(
            inps,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.generate(**inputs, **self._generation_args)

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


class CausalLMReader(HuggingFaceReader):
    _model_class = AutoModelForCausalLM


class Seq2SeqLMReader(HuggingFaceReader):
    _model_class = AutoModelForSeq2SeqLM
