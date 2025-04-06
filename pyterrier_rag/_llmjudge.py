from typing import List
import pandas as pd
import ir_measures
import pyterrier as pt

from pyterrier_rag.prompt._judge import (
    PointwiseLLMJudgePrompt,
    PointwiseLLMJudgeSystemMessage,
)
from pyterrier_rag.backend._base import Backend
from pyterrier_rag.prompt import PromptTransformer

backend_obj = None

def llmjudge_fn(prediction: str, gold: List[str], backend: Backend=None):
    """
    LLMasJudge function to evaluate the prediction against the gold standard.
    """
    global backend_obj
    if backend is None:
        if backend_obj is None:
            raise ValueError("Backend must be provided or set globally.")
        backend = backend_obj
    else:
        backend_obj = backend
    prompt = PromptTransformer(
        system_message=PointwiseLLMJudgeSystemMessage,
        instruction=PointwiseLLMJudgePrompt,
        model_name_or_path=backend.model_name_or_path,
        relevant_fields=["prediction", "gold"],
    )
    prompt_string = [prompt.create_prompt({
        "prediction": prediction,
        "gold": g,
    }) for g in gold]
    response = backend.generate(prompt_string)
    # parse to int
    rating = []
    for r in response:
        try:
            rating.append(int(r.split("[[")[1].split("]]")[0]))
        except ValueError:
            rating.append(0)
    return rating

LLMasJudge = ir_measures.define_byquery(
    lambda qrels, res, backend: max(
        [
            llmjudge_fn(res.iloc[0]["qanswer"], gold, backend)
            for gold in qrels["gold_answer"]
        ]
    ),
    support_cutoff=False,
    name="LLMasJudge",
)