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
    """
    Backend implementation using the vLLM library for text generation and sparse logits.

    Parameters:
        model_name_or_path (str): Identifier or path of the vLLM model.
        model_args (dict, optional): Keyword arguments for LLM instantiation.
        output_format (str): Desired output format (default 'text').
        generation_args (dict, optional): Parameters for sampling (e.g., max_tokens, temperature).
        batch_size (int): Prompts to process per batch (inherited).
        max_input_length (int): Maximum tokens per input prompt (inherited).
        max_new_tokens (int): Tokens to generate per prompt (inherited).
        verbose (bool): Enable verbose output.
        **kwargs: Additional parameters forwarded to the base Backend class.

    Raises:
        ImportError: If the vllm library is unavailable.
    """
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
        return_logits: bool = True,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(
            output_format=output_format,
            batch_size=batch_size,
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
            return_logits=return_logits,
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
                "max_tokens": self.max_new_tokens,
                "temperature": 1.0,
            }
        if self.return_logits:
            generation_args["logprobs"] = 20
        self.generation_args = generation_args
        self.to_params = SamplingParams

    def generate(self, inps: Iterable[str], **kwargs) -> List[BackendOutput]:
        args = self.to_params(**self.generation_args, **kwargs)
        responses = self.model.generate(inps, args)
        text = map(lambda x: x.outputs[0].text, responses)

        if self.return_logits:
            logits = map(lambda x: x.outputs[0].logprobs, responses)

            return [BackendOutput(text=txt, logits=logit) for txt, logit in zip(text, logits)]
        return [BackendOutput(text=txt) for txt in text]
