from typing import Iterable, List
import numpy as np

from pyterrier_rag.backend._base import Backend, BackendOutput
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


class VLLMBackend(Backend):
    """
    Backend implementation using the vLLM library for text generation and sparse logits.

    Parameters:
        model_name_or_path (str): Identifier or path of the vLLM model.
        model_args (dict, optional): Keyword arguments for LLM instantiation.
        generation_args (dict, optional): Parameters for sampling (e.g., max_tokens, temperature).
        max_input_length (int): Maximum tokens per input prompt (inherited).
        max_new_tokens (int): Tokens to generate per prompt (inherited).
        verbose (bool): Enable verbose output.

    Raises:
        ImportError: If the vllm library is unavailable.
    """
    _logit_type = "sparse"
    _support_logits = True

    def __init__(
        self,
        model_name_or_path: str,
        *,
        model_args: dict = {},
        generation_args: dict = None,
        max_input_length: int = 512,
        max_new_tokens: int = 32,
        verbose: bool = False,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
            verbose=verbose,
        )
        if not is_vllm_availible():
            raise ImportError("Please install vllm to use VLLMBackend")
        from vllm import LLM, SamplingParams

        self.model = LLM(model=model_name_or_path, **model_args)

        if generation_args is None:
            generation_args = {
                "max_tokens": self.max_new_tokens,
                "temperature": 1.0,
            }
        self.generation_args = generation_args
        self.to_params = SamplingParams

    def generate(self, inps: Iterable[str], return_logits=False, **kwargs) -> List[BackendOutput]:
        generation_args = {}
        generation_args.update(self.generation_args)
        generation_args.update(kwargs)
        if return_logits:
            generation_args.update({'logprobs': 20})
        args = self.to_params(**generation_args)
        responses = self.model.generate(inps, args)
        text = map(lambda x: x.outputs[0].text, responses)

        if return_logits:
            logits = map(lambda x: x.outputs[0].logprobs, responses)

            return [BackendOutput(text=txt, logits=logit) for txt, logit in zip(text, logits)]
        return [BackendOutput(text=txt) for txt in text]
