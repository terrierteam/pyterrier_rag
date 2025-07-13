import sys
import os
from typing import List, Optional, Literal, Union, Dict
from concurrent.futures import ThreadPoolExecutor

from pyterrier_rag._optional import is_openai_availible
from pyterrier_rag.backend._base import Backend, BackendOutput


class OpenAIBackend(Backend):
    """
    Backend using an OpenAI API-compatible endpoint.

    Parameters:
        model_id (str): OpenAI model identifier.
        api_key (str, optional): API key or set via OPENAI_API_KEY env var.
        generation_args (dict, optional): Params for ChatCompletion.create.
        max_input_length (int): Max prompt tokens.
        max_new_tokens (int): Max tokens to generate.
        max_retries (int): Retry attempts for API errors.
        api (str): Which API endpoint to use.
        base_url (str): Base API URL
        timeout (float): Timeout for API calls
        parallel (int): Number of parallel requests to issue to the API.
        verbose (bool): Enable verbose logging.
    """
    supports_logprobs = True
    supports_num_responses = True
    @property
    def supports_message_input(self):
        return self.api == 'chat/completions'

    def __init__(
        self,
        model_id: str,
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
        parallel: int = 4,
        verbose: bool = False,
    ):
        super().__init__(
            model_id=model_id,
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
        self.thread_pool = ThreadPoolExecutor(max_workers=parallel)
        self.api = api

    def _call_completion(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        return_logprobs: bool = False,
        num_responses: int = 1,
    ) -> List[BackendOutput]:
        if not isinstance(prompt, str):
            raise ValueError("prompt must be str when using the completions API")
        args = {
            'model': self.model_id,
            'timeout': self.timeout,
        }
        args.update(self._generation_args)
        args['n'] = num_responses
        if max_new_tokens:
            args['max_tokens'] = max_new_tokens
        if return_logprobs:
            args['logprobs'] = self.logprobs_topk
        try:
            completions = self.client.completions.create(prompt=prompt, **args)
        except Exception as e:
            sys.stderr.write(str(e) + '\n')
            if "This model's maximum context length is" in str(e):
                return [BackendOutput(text="ERROR::reduce_length")] * num_responses
            if "The response was filtered" in str(e):
                return [BackendOutput(text="ERROR::response_filtered")] * num_responses
            return [BackendOutput(text="ERROR::other")] * num_responses
        results = []
        for choice in completions.choices:
            results.append(BackendOutput(text=choice.text))
            if return_logprobs and choice.logprobs is not None:
                results[-1].logprobs = choice.logprobs.top_logprobs
        if len(results) < num_responses:
            # Fill with empty outputs if fewer responses than requested
            results += [BackendOutput(text="")] * (num_responses - len(results))
        return results

    def _call_chat_completion(
        self,
        messages: List[dict],
        max_new_tokens: Optional[int] = None,
        return_logprobs: bool = False,
        num_responses: int = 1,
    ) -> List[BackendOutput]:
        args = {
            'model': self.model_id,
            'timeout': self.timeout,
        }
        args.update(self._generation_args)
        args['n'] = num_responses
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
                return [BackendOutput(text="ERROR::reduce_length")] * args.num_responses
            if "The response was filtered" in str(e):
                return [BackendOutput(text="ERROR::response_filtered")] * args.num_responses
            return [BackendOutput(text="ERROR::other")] * num_responses
        results = []
        for choice in completions.choices[:num_responses]:
            results.append(BackendOutput(text=choice.message.content))
            if return_logprobs and choice.logprobs is not None:
                results[-1].logprobs = [{lp.token: lp.logprob for lp in lps.top_logprobs} for lps in choice.logprobs.content]
        if len(results) < num_responses:
            # Fill with empty outputs if fewer responses than requested
            results += [BackendOutput(text="")] * (num_responses - len(results))
        return results

    def generate(
        self,
        inps: Union[List[str], List[List[dict]]],
        *,
        return_logprobs: bool = False,
        max_new_tokens: Optional[int] = None,
        num_responses: int = 1,
    ) -> List[BackendOutput]:
        futures = []
        if self.api == 'completions':
            for inp in inps:
                futures.append(self.thread_pool.submit(
                    self._call_completion,
                    inp,
                    max_new_tokens=max_new_tokens,
                    return_logprobs=return_logprobs,
                    num_responses=num_responses,
                ))
        elif self.api == 'chat/completions':
            for inp in inps:
                if isinstance(inp, str):
                    # treat plain str inputs as simple messages
                    inp = [{"role": "user", "content": inp}]
                futures.append(self.thread_pool.submit(
                    self._call_chat_completion,
                    inp,
                    max_new_tokens=max_new_tokens,
                    return_logprobs=return_logprobs,
                    num_responses=num_responses,
                ))
        else:
            raise ValueError(f'api {self.api!r} not supported')
        results = []
        for r in futures:
            results.extend(r.result())
        return results

    @staticmethod
    def from_params(params: Dict[str, str]) -> 'OpenAIBackend':
        """Create an OpenAIBackend instance from the provided parameters.

        Supported params:

            - model_id: str, the OpenAI model identifier (required)
            - api_key: str, API key for OpenAI (default: None, uses OPENAI_API_KEY env var). If value starts with $, loads the value from the provided environment variable.
            - max_retries: int, number of retries for API errors (default: 10)
            - base_url: str, base URL for the OpenAI API (default: None)
            - timeout: float, timeout for API calls in seconds (default: 30.0)
            - logprobs_topk: int, number of top log probabilities to return (default: 20)
            - parallel: int, number of parallel requests to issue to the API (default: 4)
            - verbose: bool, enable verbose logging (default: False)
        
            Returns:
                OpenAIBackend: An instance of OpenAIBackend.
        """
        api_key = params.get("api_key")
        if api_key and api_key.startswith("$"):
            env_var = api_key[1:]
            api_key = os.environ.get(env_var)
            if not api_key:
                raise ValueError(f"Environment variable {env_var} not found for OpenAI API key")

        return OpenAIBackend(
            model_id=params["model_id"],
            api_key=api_key,
            max_retries=int(params.get("max_retries", 10)),
            base_url=params.get("base_url"),
            timeout=float(params.get("timeout", 30.0)),
            logprobs_topk=int(params.get("logprobs_topk", 20)),
            parallel=int(params.get("parallel", 4)),
            verbose=bool(params.get("verbose", False)),
        )

    def __repr__(self):
        return f"OpenAIBackend({self.model_id!r})"

    def __del__(self):
        if hasattr(self, 'thread_pool') and self.thread_pool is not None:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None


__all__ = ["OpenAIBackend"]
