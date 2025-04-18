from enum import Enum
from pyterrier_rag.backend._base import Backend


class Backends(Enum):
    HF = "hf"
    VBackend = "vBackend"
    OPENAI = "openai"


def get_backend(backend_type: str, model_name: str, **Backend_kwargs) -> Backend:
    """
    Get the backend object based on the backend type and model name.
    """
    if backend_type == Backends.HF.value:
        from pyterrier_rag.backend._hf import HuggingFaceBackend

        return HuggingFaceBackend(model_name, **Backend_kwargs)
    elif backend_type == Backends.VBackend.value:
        from pyterrier_rag.backend._vllm import VLLMBackend

        return VLLMBackend(model_name, **Backend_kwargs)
    elif backend_type == Backends.OPENAI.value:
        from pyterrier_rag.backend._openai import OpenAIBackend

        return OpenAIBackend(model_name, **Backend_kwargs)
    else:
        raise ValueError(f"Unknown Backend type: {backend_type}")


__all__ = ["BackendS", "get_Backend"]
