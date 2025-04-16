from enum import Enum
from pyterrier_rag.backend._base import Backend


class BACKENDS(Enum):
    HF = "hf"
    VLLM = "vllm"
    OPENAI = "openai"


def get_backend(backend_type: str, model_name: str, **backend_kwargs) -> Backend:
    """
    Get the backend object based on the backend type and model name.
    """
    if backend_type == BACKENDS.HF.value:
        from pyterrier_rag.backend._hf import HuggingFaceBackend

        return HuggingFaceBackend(model_name, **backend_kwargs)
    elif backend_type == BACKENDS.VLLM.value:
        from pyterrier_rag.backend._vllm import VLLMBackend

        return VLLMBackend(model_name, **backend_kwargs)
    elif backend_type == BACKENDS.OPENAI.value:
        from pyterrier_rag.backend._openai import OpenAIBackend

        return OpenAIBackend(model_name, **backend_kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


__all__ = ["BACKENDS", "get_backend"]
