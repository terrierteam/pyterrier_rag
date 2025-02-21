from .prompt._judge import (
    PairwiseLLMJudgePrompt,
    PointwiseLLMJudgePrompt,
    PairwiseLLMJudgeSystemMessage,
    PointwiseLLMJudgeSystemMessage,
)
from .prompt import PromptTransformer


def _pointwise_llm_judge(model):
    prompt = PromptTransformer(
        system_message=PointwiseLLMJudgeSystemMessage,
        instruction=PointwiseLLMJudgePrompt,
        model_name_or_path=model.model_name_or_path,
        relevant_fields=["prediction", "gold"],
    )
    return prompt >> model


def _pairwise_llm_judge(model):
    prompt = PromptTransformer(
        system_message=PairwiseLLMJudgeSystemMessage,
        instruction=PairwiseLLMJudgePrompt,
        model_name_or_path=model.model_name_or_path,
        relevant_fields=["prediction", "gold"],
    )
    return prompt >> model


def _make_pairwise_judge_function(model: pt.Transformer, minlabel: int = 3):
    judge = _pairwise_llm_judge(model)

    def judge_fn(res: pd.DataFrame, qrels: pd.DataFrame):
        import pyterrier_alpha as pta

        pta.validate.result_frame(qrels, extra_columns=["label", "text"])
        pta.validate.result_frame(res, extra_columns=["text"])

        qrels = qrels[qrels.label >= minlabel]
        qrels = qrels[qrels.label >= minlabel]
        assert len(qrels), "No qrels found with minlabel of %d" % minlabel


def _make_pointwise_judge_function(model: pt.Transformer, minlabel: int = 3):
    judge = _pointwise_llm_judge(model)

    def judge_fn(res: pd.DataFrame, qrels: pd.DataFrame = None):
        import pyterrier_alpha as pta

        pta.validate.result_frame(res, extra_columns=["text"])
