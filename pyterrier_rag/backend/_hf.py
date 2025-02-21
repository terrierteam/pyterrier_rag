from typing import Iterable, List

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import torch

from ._base import Backend, BackendOutput


class HuggingFaceBackend(Backend):
    _model_class = None
    _support_logits = True
    _logit_type = "dense"

    def __init__(
        self,
        model_name_or_path: str,
        model_args: dict = {},
        output_format:str = "text",
        generation_args: dict = None,
        batch_size: int = 4,
        max_input_length: int = None,
        max_new_tokens: int = 32,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(
            output_format=output_format,
            batch_size=batch_size,
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

        max_position_embeddings = getattr(self._model.config, "max_position_embeddings", None)
        self.max_input_length = (
            max_input_length or max_position_embeddings
        )

        if generation_args is None:
            generation_args = {
                "max_new_tokens": self.max_new_tokens,
                "temperature": 1.0,
                "do_sample": False,
                "num_beams": 1,
            }
        self._generation_args = generation_args
        self.model = self._model

    @torch.no_grad()
    def generate(self, inps: Iterable[str]) -> List[str]:
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
        logits = [output.logits.numpy() for output in outputs]
        texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return [BackendOutput(text=text, logits=logit) for text, logit in zip(texts, logits)]


class CausalLMBackend(HuggingFaceBackend):
    _model_class = AutoModelForCausalLM


class Seq2SeqLMBackend(HuggingFaceBackend):
    _model_class = AutoModelForSeq2SeqLM
