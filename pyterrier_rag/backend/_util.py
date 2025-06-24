from enum import Enum
from pyterrier_rag.backend._base import Backend
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


__all__ = ["Backends", "get_backend"]
