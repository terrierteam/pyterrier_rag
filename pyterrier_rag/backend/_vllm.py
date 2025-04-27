from typing import Iterable, List
import numpy as np

from pyterrier_rag.backend._base import Backend as ragBackend, BackendOutput
from pyterrier_rag._optional import is_vllm_availible


def get_logits_from_dict(d: List[dict], tokenizer):
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


class VLLMBackend(ragBackend):
    _logit_type = "sparse"
    _support_logits = True
    _remove_prompt = True

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
            raise ImportError("Please install vllm to use VLLMBackend")
        from vllm import LLM, SamplingParams

        self._model_name_or_path = model_name_or_path
        self.model = LLM(model=model_name_or_path, **model_args)

        if generation_args is None:
            generation_args = {
                "max_new_tokens": self.max_new_tokens,
                "temperature": 1.0,
                "do_sample": False,
                "num_beams": 1,
            }
        generation_args["log_probs"] = llm.get_tokenizer().vocab_size
        self.generation_args = generation_args
        self.to_params = lambda x: SamplingParams(**x)

    def generate(self, inps: Iterable[str], **kwargs) -> Iterable[str]:
        args = self.to_params(**self.generation_args, **kwargs)
        responses = self.model.generate(inps, args)
        logits = map(lambda x: x[0].log_probs, responses)
        text = map(lambda x: x[0].text, responses)

        return [BackendOutput(text=t, logits=l) for t, l in zip(text, logits)]
