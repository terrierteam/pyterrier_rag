import os
import time
from typing import List

from .._optional import is_openai_availible, is_tiktoken_availible
from ._base import Reader


class OpenAIReader(Reader):
    def __init__(
        self,
        model_name_or_path: str,
        api_key: str = None,
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
        if self.output_format != "text":
            raise ValueError("OpenAIReader currently only supports output_format='text'")
        if not is_openai_availible():
            raise ImportError("Please install openai to use OpenAIReader")
        import openai

        self._key = api_key or os.environ.get("OPENAI_API_KEY")
        if self._key is None:
            raise ValueError(
                "api_key must be provided or set as an environment variable OPENAI_API_KEY"
            )
        openai.api_key = self._key
        self._model_name_or_path = model_name_or_path
        if is_tiktoken_availible():
            import tiktoken

            self._tokenizer = tiktoken.get_encoding(self.model)
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

    @property
    def is_openai(self):
        return True

    def _call_completion(
        self,
        *args,
        return_text=False,
        **kwargs,
    ) -> List[int]:
        import openai

        while True:
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
        if return_text:
            completion = completion["choices"][0]["message"]["content"]
        return completion

    def _generate(self, prompt: List[dict]) -> List[str]:
        response = self._call_completion(
            messages=prompt,
            return_text=True,
            **{"model": self._model_name_or_path, **self._generation_args},
        )
        return response


__all__ = ["OpenAIReader"]
