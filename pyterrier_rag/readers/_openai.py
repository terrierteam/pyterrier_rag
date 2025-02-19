import os
import time
from typing import Any, Iterable, List

from .._optional import is_openai_availible, is_tiktoken_availible
from . import _content_aggregation as content_aggregation
from ._base import GENERIC_PROMPT, Reader


class OpenAIReader(Reader):
    _prompt = GENERIC_PROMPT
    def __init__(self,
                 model_name_or_path: str,
                 api_key : str = None,
                 system_message : str = None,
                 generation_args : dict = None,
                 context_aggregation : str = 'concat',
                 prompt : Any = None,
                 batch_size: int = 4,
                 text_field: str = 'text',
                 text_max_length: int = 512,
                 num_context: int = 5,
                 max_new_tokens: int = 32,
                 verbose: bool = False,
                 **kwargs
                ):
        super().__init__(batch_size=batch_size,
                         text_field=text_field,
                         text_max_length=text_max_length,
                         num_context=num_context,
                         max_new_tokens=max_new_tokens,
                         generation_config=None,
                         verbose=verbose,
                         **kwargs)
        if not is_openai_availible():
            raise ImportError("Please install openai to use OpenAIReader")
        import openai
        self._key = api_key or os.environ.get("OPENAI_API_KEY")
        if self._key is None:
            raise ValueError("api_key must be provided or set as an environment variable OPENAI_API_KEY")
        openai.api_key = self._key
        self._model_name_or_path = model_name_or_path
        if is_tiktoken_availible():
            import tiktoken
            self._tokenizer = tiktoken.get_encoding(self.model)
        else:
            self._tokenizer = None

        if context_aggregation not in content_aggregation.__all__:
            raise ValueError(f"context_aggregation must be one of {content_aggregation.__all__}")
        self._context_aggregation = getattr(content_aggregation, context_aggregation)
        self._prompt = prompt or self._prompt
        self._system_message = system_message

        if isinstance(self._prompt, str):
            self._prompt = self._prompt.format

        if generation_args is None:
            generation_args = {
                'max_new_tokens' : self.max_new_tokens,
                'temperature' : 1.0,
                'do_sample' : False,
                'num_beams' : 1,
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
                completion = openai.ChatCompletion.create(
                    *args, **kwargs, timeout=30
                )
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
            completion = (
                completion["choices"][0]["message"]["content"]
            )
        return completion

    def _generate(self, prompt : str):
        messages = []
        if self._system_message:
            messages.append(
                {"role": "system", "content": self._system_message}
            )
        messages.append(
            {"role": "user", "content": prompt}
        )
        response = self._call_completion(
            messages=messages,
            return_text=True,
            **{'model': self._model_name_or_path, **self._generation_args},
        )
        return response

    def transform_by_query(self, inp: Iterable[dict]) -> Iterable[dict]:
        inp = list(inp)
        qid = inp[0]["qid"]
        query = inp[0]["query"]

        context = self.get_context_by_query(inp)
        if self._tokenizer is None:
            aggregate_context = self._context_aggregation(context)
        else:
            aggregate_context = self._context_aggregation(context,
                                                          tokenizer=self._tokenizer,
                                                          max_length=self.text_max_length-len(self._tokenizer.encode(query)))
        input_texts = self._prompt(query=query, context=aggregate_context)
        outputs = self._generate(input_texts)

        return [{"qid": qid, "query": query, "qanswer": outputs}]
