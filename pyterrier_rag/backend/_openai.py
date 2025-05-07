import os
import time
from typing import List

from pyterrier_rag._optional import is_openai_availible, is_tiktoken_availible
from pyterrier_rag.backend._base import Backend, BackendOutput


class OpenAIBackend(Backend):
    """
    Backend using OpenAI ChatCompletion.

    Parameters:
        model_name_or_path (str): OpenAI model identifier.
        api_key (str, optional): API key or set via OPENAI_API_KEY env var.
        generation_args (dict, optional): Params for ChatCompletion.create.
        batch_size (int): Prompts per batch.
        max_input_length (int): Max prompt tokens.
        max_new_tokens (int): Max tokens to generate.
        max_trials (int): Retry attempts for API errors.
        verbose (bool): Enable verbose logging.
        **kwargs: Passed to Backend base class.
    """
    _api_type = "openai"

    def __init__(
        self,
        model_name_or_path: str,
        api_key: str = None,
        generation_args: dict = None,
        batch_size: int = 4,
        max_input_length: int = 512,
        max_new_tokens: int = 32,
        return_logits: bool = False,
        max_trials: int = 10,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(
            batch_size=batch_size,
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
            generation_config=None,
            return_logits=return_logits,
            verbose=verbose,
            **kwargs,
        )
        if not is_openai_availible():
            raise ImportError("Please install openai to use OpenAIBackend")
        import openai

        self._key = api_key or os.environ.get("OPENAI_API_KEY")
        if self._key is None:
            raise ValueError("api_key must be provided or set as an environment variable OPENAI_API_KEY")
        openai.api_key = self._key
        self._model_name_or_path = model_name_or_path
        if is_tiktoken_availible():
            import tiktoken

            self._tokenizer = tiktoken.encoding_for_model(self._model_name_or_path)
        else:
            self._tokenizer = None

        if generation_args is None:
            generation_args = {
                "max_new_tokens": self.max_new_tokens,
                "temperature": 1.0,
                "do_sample": False,
                "num_beams": 1,
            }
        self._generation_args = generation_args
        self.max_trials = max_trials

    def _call_completion(
        self,
        *args,
        return_text=False,
        **kwargs,
    ) -> List[int]:
        import openai

        trials = self.max_trials
        while True:
            if trials <= 0:
                print(f"Exceeded {self.max_trials}, exiting")
                if return_text:
                    return ""
                return {}
            try:
                completion = openai.ChatCompletion.create(*args, **kwargs, timeout=30)
                break
            except Exception as e:
                print(str(e))
                if "This model's maximum context length is" in str(e):
                    print("reduce_length")
                    return "ERROR::reduce_length"
                if "The response was filtered" in str(e):
                    print("The response was filtered")
                    return "ERROR::The response was filtered"
                time.sleep(0.1)
                trials -= 1
        if return_text:
            completion = completion["choices"][0]["message"]["content"]
        return completion

    def generate(self, prompt: List[dict], **kwargs) -> List[BackendOutput]:
        response = self._call_completion(
            messages=prompt,
            return_text=True,
            **{"model": self._model_name_or_path, **self._generation_args, **kwargs},
        )
        return [BackendOutput(text=r) for r in response]


__all__ = ["OpenAIBackend"]
