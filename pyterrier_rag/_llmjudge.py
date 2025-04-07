from typing import List
import pandas as pd
import ir_measures
import pyterrier as pt
import pyterrier_alpha as pta

from pyterrier_rag.prompt._judge import (
    PointwiseLLMJudgePrompt,
    PointwiseLLMJudgeSystemMessage,
)
from pyterrier_rag.backend._base import Backend
from pyterrier_rag.prompt import PromptTransformer

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
    

backend_obj, prompt = None, None

def llmjudge_fn(qrels, res, backend_type: str, model_name: str, rel = 3, agg = 'max') -> int:
    """
    LLMasJudge function to evaluate the prediction against the gold standard.
    """

    pta.validate.columns(qrels, includes=['query_id', 'relevance', 'text'])
    pta.validate.columns(res, includes=['query_id', 'qanswer'])
    
    assert len(res), "Empty res df provided"
    assert len(qrels), "Empty qrels df provided"
    qrels = qrels[qrels.relevance >= rel]
    assert len(qrels), "No qrels found with minimum label of %d" % rel

    global backend_obj
    global prompt
    if backend_obj is None:
        backend_obj = get_backend(backend_type, model_name)
        prompt = PromptTransformer(
            PointwiseLLMJudgePrompt,
            backend_obj.model_name_or_path
            system_message=PointwiseLLMJudgeSystemMessage,
        )

    predictions = res['qanswer'].to_list()
    assert len(predictions) == 1, "Unexpected number of predictions"
    references = qrels['text'].to_list()
    # duplicate the prediction for the nbr of ground truths 
    predictions = predictions * len(references)

    # create the prompt
    prompts = [prompt.create_prompt(
        {
            'prediction': p,
            'reference': r,
        }
    ) for p, r in zip(predictions, references)]

    outputs = backend_obj.generate(prompts)
    parsed_ints = []
    for output in outputs:
        parsed_ints.append(int(output.split()[0]))
    assert len(parsed_ints) == len(outputs), "Unexpected number of parsed integers"
    assert len(parsed_ints) == len(predictions), "Unexpected number of parsed integers"
    # aggregate the scores
    if agg == 'max':
        return max(parsed_ints)
    elif agg == 'avg':
        return sum(parsed_ints) / len(parsed_ints)
    elif agg == 'sum':
        return sum(parsed_ints)
    elif agg == 'min':
        return min(parsed_ints)
    elif agg == 'none':
        return parsed_ints
    else:
        raise ValueError(f"Unknown aggregation method: {agg}")


def LLMasJudge(backend_type, model_name_or_path):
    return ir_measures.define_byquery(
        lambda qrels, res: llmjudge_fn(qrels, res, backend_type=backend_type, model_name=model_name_or_path),
        name='LLMasJudge',
        support_cutoff=False,
    )