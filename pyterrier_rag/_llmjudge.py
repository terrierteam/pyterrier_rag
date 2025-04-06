import pandas as pd
import ir_measures
import pyterrier as pt

from pyterrier_rag.prompt._judge import (
    PairwiseLLMJudgePrompt,
    PointwiseLLMJudgePrompt,
    PairwiseLLMJudgeSystemMessage,
    PointwiseLLMJudgeSystemMessage,
)
from pyterrier_rag.backend._base import Backend
from pyterrier_rag.prompt import PromptTransformer


def _pointwise_llm_judge(backend):
    prompt = PromptTransformer(
        system_message=PointwiseLLMJudgeSystemMessage,
        instruction=PointwiseLLMJudgePrompt,
        model_name_or_path=backend.model_name_or_path,
        relevant_fields=["prediction", "gold"],
    )
    return prompt >> pt.apply.generic(lambda x: backend.generate(x['query']))


def _pairwise_llm_judge(backend):
    prompt = PromptTransformer(
        system_message=PairwiseLLMJudgeSystemMessage,
        instruction=PairwiseLLMJudgePrompt,
        model_name_or_path=backend.model_name_or_path,
        relevant_fields=["prediction", "gold"],
    )
    return prompt >> pt.apply.generic(lambda x: backend.generate(x['query']))


def _make_pairwise_judge_function(backend: Backend, minlabel: int = 3):
    judge = _pairwise_llm_judge(backend)

    def judge_fn(res: pd.DataFrame, qrels: pd.DataFrame):
        import pyterrier_alpha as pta

        pta.validate.result_frame(qrels, extra_columns=["label", "text"])
        pta.validate.result_frame(res, extra_columns=["text"])

        qrels = qrels[qrels.label >= minlabel]
        qrels = qrels[qrels.label >= minlabel]
        assert len(qrels), "No qrels found with minlabel of %d" % minlabel


def _make_pointwise_judge_function(backend: Backend, minlabel: int = 3):
    judge = _pointwise_llm_judge(backend)

    def judge_fn(res: pd.DataFrame, qrels: pd.DataFrame = None):
        import pyterrier_alpha as pta

        pta.validate.result_frame(res, extra_columns=["text"])
        qrels = qrels[qrels.label >= minlabel]
        qrels = qrels[qrels.label >= minlabel]
        assert len(qrels), "No qrels found with minlabel of %d" % minlabel

LLMJudge = ir_measures.define_byquery(
    lambda qrels, res: max(
        [judge(res.iloc[0], qrels) for judge in _make_pairwise_judge_function()]
    ),
    support_cutoff=False,
    name="LLMJudge",
)