import os
import sys
from typing import Union, List, Optional
from enum import Enum
from pyterrier_rag.backend._base import Backend, BackendOutput
from pyterrier_rag.backend._hf import HuggingFaceBackend
from pyterrier_rag._optional import is_vllm_availible, is_openai_availible


class Backends(Enum):
    HF_CAUSAL = "causal"
    HF_SEQ2SEQ = "seq2seq"
    VLLM = "vllm"
    OPENAI = "openai"


def get_backend(backend_type: str, model_name: str, **backend_kwargs) -> Backend:
    """
    Get the backend object based on the backend type and model name.
    """
    if backend_type == Backends.HF_CAUSAL.value:

        return HuggingFaceBackend(model_name, **backend_kwargs)
    if backend_type == Backends.HF_SEQ2SEQ.value:

        return HuggingFaceBackend(model_name, **backend_kwargs)
    elif backend_type == Backends.VLLM.value:
        if not is_vllm_availible():
            raise ImportError("Install vLLM to use VLLMBackend")
        from pyterrier_rag.backend._vllm import VLLMBackend

        return VLLMBackend(model_name, **backend_kwargs)
    elif backend_type == Backends.OPENAI.value:
        if not is_openai_availible():
            raise ImportError("Install openai to use OpenAIBackend")
        from pyterrier_rag.backend._openai import OpenAIBackend

        return OpenAIBackend(model_name, **backend_kwargs)
    else:
        raise ValueError(f"Unknown Backend type: {backend_type}")


class _DefaultBackend(Backend):
    def __init__(self):
        self._backend = None

    @property
    def backend(self):
        if self._backend is None:
            raise RuntimeError("You need to run default_backend.set(backend) before using default_backend.")
        return self._backend

    @property
    def supports_logprobs(self):
        return self.backend.supports_logprobs

    @property
    def supports_message_input(self):
        return self.backend.supports_message_input

    @property
    def supports_num_responses(self):
        return self.backend.supports_num_responses

    @property
    def model_id(self):
        return self.backend.model_id

    @property
    def max_input_length(self):
        return self.backend.max_input_length

    @property
    def max_new_tokens(self):
        return self.backend.max_new_tokens

    @property
    def verbose(self):
        return self.backend.verbose

    def set(self, backend: Backend, *, verbose=False):
        """ Set the default backend to use for text generation.
        
        Parameters:
            backend (Backend): The backend instance to set.
        """
        if verbose:
            if self._backend is None:
                sys.stderr.write(f"set default backend to {backend!r}\n")
            else:
                sys.stderr.write(f"replaced default backend {self._backend!r} with {backend!r}\n")
        self._backend = backend

    def generate(
        self,
        inps: Union[List[str], List[List[dict]]],
        *,
        return_logprobs: bool = False,
        max_new_tokens: Optional[int] = None,
        num_responses: int = 1,
    ) -> List[BackendOutput]:
        """ Delegate the generation to the set backend. """
        return self.backend.generate(inps, return_logprobs=return_logprobs, max_new_tokens=max_new_tokens, num_responses=num_responses)

    def __repr__(self):
        if self._backend is None:
            return "<DefaultBackend: not set>"
        return repr(self._backend)


default_backend = _DefaultBackend()
if os.environ.get('PYTERRIER_RAG_DEFAULT_BACKEND'):
    try:
        default_backend.set(Backend.from_dsn(os.environ["PYTERRIER_RAG_DEFAULT_BACKEND"]), verbose=False)
        sys.stderr.write(f'set default_backend to {default_backend.backend} from PYTERRIER_RAG_DEFAULT_BACKEND\n')
    except ValueError as ex:
        sys.stderr.write(f'error setting default_backend from PYTERRIER_RAG_DEFAULT_BACKEND={os.environ["PYTERRIER_RAG_DEFAULT_BACKEND"]!r}: {ex}\n')

__all__ = ["Backends", "get_backend", "default_backend"]
