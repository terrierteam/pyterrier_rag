from enum import Enum
from pyterrier_rag.llm._base import LLM


class LLMS(Enum):
    HF = "hf"
    VLLM = "vllm"
    OPENAI = "openai"


def get_LLM(LLM_type: str, model_name: str, **LLM_kwargs) -> LLM:
    """
    Get the LLM object based on the LLM type and model name.
    """
    if LLM_type == LLMS.HF.value:
        from pyterrier_rag.llm._hf import HuggingFaceLLM

        return HuggingFaceLLM(model_name, **LLM_kwargs)
    elif LLM_type == LLMS.VLLM.value:
        from pyterrier_rag.llm._vllm import VLLMLLM

        return VLLMLLM(model_name, **LLM_kwargs)
    elif LLM_type == LLMS.OPENAI.value:
        from pyterrier_rag.llm._openai import OpenAILLM

        return OpenAILLM(model_name, **LLM_kwargs)
    else:
        raise ValueError(f"Unknown LLM type: {LLM_type}")


__all__ = ["LLMS", "get_LLM"]
