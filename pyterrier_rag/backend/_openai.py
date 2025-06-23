import os
from typing import List, Iterable, Literal

from pyterrier_rag._optional import is_openai_availible, is_tiktoken_availible
from pyterrier_rag.backend._base import Backend, BackendOutput


class OpenAIBackend(Backend):
    """
    Backend using OpenAI API endpoint(s).

    Parameters:
        model_name_or_path (str): OpenAI model identifier.
        api_key (str, optional): API key or set via OPENAI_API_KEY env var.
        generation_args (dict, optional): Params for ChatCompletion.create.
        batch_size (int): Prompts per batch.
        max_input_length (int): Max prompt tokens.
        max_new_tokens (int): Max tokens to generate.
        max_retries (int): Retry attempts for API errors.
        api (str): Which API endpoint to use.
        base_url (str): Base API URL
        timeout (float): Timeout for API calls
        verbose (bool): Enable verbose logging.
        **kwargs: Passed to Backend base class.
    """
    _api_type = "openai"

    def __init__(
        self,
        model_name_or_path: str,
        *,
        api_key: str = None,
        generation_args: dict = None,
        batch_size: int = 4,
        max_input_length: int = 512,
        max_new_tokens: int = 32,
        return_logits: bool = False,
        max_retries: int = 10,
        api: Literal['chat/completions', 'completions'] = 'chat/completions',
        base_url: str = None,
        timeout: float = 30.,
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

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("api_key must be provided or set as an environment variable OPENAI_API_KEY")
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=max_retries,
        )
        self._model_name_or_path = model_name_or_path
        if is_tiktoken_availible():
            import tiktoken

            try:
                self._tokenizer = tiktoken.encoding_for_model(self._model_name_or_path)
            except KeyError:
                self._tokenizer = None
        else:
            self._tokenizer = None

        if generation_args is None:
            generation_args = {
                "max_tokens": self.max_new_tokens,
                "temperature": 1.0,
            }
        self._generation_args = generation_args
        self.timeout = timeout
        self.api = api

    def _call_completion(
        self,
        prompts,
        return_text=False,
        **kwargs,
    ) -> List[int]:
        args = {
            'model': self._model_name_or_path,
            'timeout': self.timeout,
        }
        args.update(self._generation_args)
        args.update(kwargs)
        try:
            completions = self.client.completions.create(prompt=prompts, **args)
        except Exception as e:
            print(str(e))
            if "This model's maximum context length is" in str(e):
                print("reduce_length")
                return ["ERROR::reduce_length" for _ in prompts]
            if "The response was filtered" in str(e):
                print("The response was filtered")
                return ["ERROR::The response was filtered" for _ in prompts]
            return ['ERROR:other' for _ in prompts]
        if return_text:
            completions = [c.text for c in completions.choices]
        return completions

    def _call_chat_completion(
        self,
        messages,
        return_text=False,
        **kwargs,
    ) -> List[int]:
        args = {
            'model': self._model_name_or_path,
            'timeout': self.timeout,
        }
        args.update(self._generation_args)
        args.update(kwargs)
        try:
            completions = self.client.chat.completions.create(messages=messages, **args)
        except Exception as e:
            print(str(e))
            if "This model's maximum context length is" in str(e):
                print("reduce_length")
                return "ERROR::reduce_length"
            if "The response was filtered" in str(e):
                print("The response was filtered")
                return "ERROR::The response was filtered"
            return 'ERROR:other'
        if return_text:
            completions = completions.choices[0].message.content
        return completions

    def generate(self, inps: Iterable[str], max_new_tokens=None, **kwargs) -> List[BackendOutput]:
        inps = list(inps)
        if max_new_tokens is not None:
            kwargs['max_tokens'] = max_new_tokens
        if self.api == 'completions':
            responses = self._call_completion(
                inps,
                return_text=True,
                **kwargs,
            )
        elif self.api == 'chat/completions':
            responses = []
            for inp in inps:
                responses.append(self._call_chat_completion(
                    [{"role": "user", "content": inp} for inp in inps],
                    return_text=True,
                    **kwargs,
                ))
        else:
            raise ValueError(f'api {self.api!r} not supported')
        return [BackendOutput(text=r) for r in responses]


__all__ = ["OpenAIBackend"]
