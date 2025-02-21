from typing import Iterable, List
import numpy as np

from ._base import Reader, ReaderOutput
from .._optional import is_vllm_availible


def get_logits_from_dict(d : List[dict], tokenizer):
    # get ordering of vocabulary from tokenizer
    vocab = tokenizer.get_vocab()
    id2token = {k: v for k, v in vocab.items()}

    # get the logits from the dictionary
    output = np.zeros((len(d), len(vocab)))
    for i in range(len(d)):
        for j in range(len(vocab)):
            # get jth token from vocab
            token = id2token[j]
            output[i, j] = d[i].get(token, 0.0)
    return output


class VLLMReader(Reader):
    _logit_type = "sparse"
    _support_logits = True

    def __init__(
        self,
        model_name_or_path: str,
        model_args: dict = {},
        output_format: str = "text",
        generation_args: dict = None,
        batch_size: int = 4,
        max_input_length: int = 512,
        max_new_tokens: int = 32,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(
            output_format=output_format,
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
        generation_args['log_probs'] = self._model.llm_engine.model_config.vocab_size
        self._generation_args = SamplingParams(**generation_args)
        self.model = LLM(self._model, self._generation_args)

    def generate(self, inps: Iterable[str]) -> Iterable[str]:
        responses = self.model.generate(inps, self._generation_args)
        logits = map(lambda x: x[0].log_probs, responses)
        text = map(lambda x: x[0].text, responses)
        return [ReaderOutput(text=t, logits=l) for t, l in zip(text, logits)]
