import os
from typing import List, Optional, Literal

from pyterrier_rag._optional import is_openai_availible
from pyterrier_rag.backend._base import Backend, BackendOutput


class OpenAIBackend(Backend):
    """
    Backend using OpenAI API endpoint(s).

    Parameters:
        model_name_or_path (str): OpenAI model identifier.
        api_key (str, optional): API key or set via OPENAI_API_KEY env var.
        generation_args (dict, optional): Params for ChatCompletion.create.
        max_input_length (int): Max prompt tokens.
        max_new_tokens (int): Max tokens to generate.
        max_retries (int): Retry attempts for API errors.
        api (str): Which API endpoint to use.
        base_url (str): Base API URL
        timeout (float): Timeout for API calls
        verbose (bool): Enable verbose logging.
    """
    supports_logprobs = True
    _api_type = "openai"

    def __init__(
        self,
        model_name_or_path: str,
        *,
        api_key: str = None,
        generation_args: dict = None,
        max_input_length: int = 512,
        max_new_tokens: int = 32,
        max_retries: int = 10,
        api: Literal['chat/completions', 'completions'] = 'chat/completions',
        base_url: str = None,
        timeout: float = 30.,
        logprobs_topk: int = 20,
        verbose: bool = False,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
            verbose=verbose,
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

        if generation_args is None:
            generation_args = {
                "max_tokens": self.max_new_tokens,
                "temperature": 1.0,
            }
        self._generation_args = generation_args
        self.timeout = timeout
        self.logprobs_topk = logprobs_topk
        self.api = api

    def _call_completion(
        self,
        prompts,
        max_new_tokens=None,
        return_logprobs: bool = False,
    ) -> List[int]:
        args = {
            'model': self.model_name_or_path,
            'timeout': self.timeout,
        }
        args.update(self._generation_args)
        if max_new_tokens:
            args['max_tokens'] = max_new_tokens
        if return_logprobs:
            args['logprobs'] = self.logprobs_topk
        try:
            completions = self.client.completions.create(prompt=prompts, **args)
        except Exception as e:
            print(str(e))
            if "This model's maximum context length is" in str(e):
                return [BackendOutput(text="ERROR::reduce_length")] * len(prompts)
            if "The response was filtered" in str(e):
                return [BackendOutput(text="ERROR::response_filtered")] * len(prompts)
            return [BackendOutput(text="ERROR::other")] * len(prompts)
        results = []
        for choice in completions.choices:
            results.append(BackendOutput(text=choice.text))
            if return_logprobs and choice.logprobs is not None:
                results[-1].logprobs = choice.logprobs.top_logprobs
        return results

    def _call_chat_completion(
        self,
        messages,
        max_new_tokens=None,
        return_logprobs: bool = False,
    ) -> List[int]:
        args = {
            'model': self.model_name_or_path,
            'timeout': self.timeout,
        }
        args.update(self._generation_args)
        if max_new_tokens:
            args['max_tokens'] = max_new_tokens
        if return_logprobs:
            args['logprobs'] = True
            args['top_logprobs'] = self.logprobs_topk
        try:
            completions = self.client.chat.completions.create(messages=messages, **args)
        except Exception as e:
            print(str(e))
            if "This model's maximum context length is" in str(e):
                return BackendOutput(text="ERROR::reduce_length")
            if "The response was filtered" in str(e):
                return BackendOutput(text="ERROR::response_filtered")
            return BackendOutput(text="ERROR::other")
        result = BackendOutput(text=completions.choices[0].message.content)
        if return_logprobs and completions.choices[0].logprobs is not None:
            result.logprobs = [{lp.token: lp.logprob for lp in lps.top_logprobs} for lps in completions.choices[0].logprobs.content]
        return result

    def generate(self,
        inps: List[str],
        *,
        return_logprobs: bool = False,
        max_new_tokens: Optional[int] = None,
    ) -> List[BackendOutput]:
        pass

        if self.api == 'completions':
            if return_logprobs:
                results = []
                for inp in inps:
                    results.append(self._call_completion(
                        [inp],
                        max_new_tokens=max_new_tokens,
                        return_logprobs=return_logprobs,
                    )[0])
            else:
                results = self._call_completion(
                    inps,
                    max_new_tokens=max_new_tokens,
                    return_logprobs=return_logprobs,
                )
        elif self.api == 'chat/completions':
            results = []
            for inp in inps:
                results.append(self._call_chat_completion(
                    [{"role": "user", "content": inp}],
                    max_new_tokens=max_new_tokens,
                    return_logprobs=return_logprobs,
                ))
        else:
            raise ValueError(f'api {self.api!r} not supported')
        return results


__all__ = ["OpenAIBackend"]
