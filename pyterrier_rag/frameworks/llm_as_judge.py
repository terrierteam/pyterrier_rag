from outlines import prompt
import ir_measures
import pyterrier_alpha as pta

from pyterrier_rag.backend import get_LLM
from pyterrier_rag.prompt import PromptTransformer

PairwiseLLMJudgeSystemMessage = 'Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user\'s instructions and answers the user\'s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.'


@prompt
def PairwiseLLMJudgePrompt(prediction: str, gold: str) -> str:
    """
    [User Question]
    {question}

    [The Start of Assistant A's Answer]
    {prediction}
    [The End of Assistant A's Answer]

    [The Start of Assistant B's Answer]
    {gold}
    [The End of Assistant B's Answer]"
    """


PointwiseLLMJudgeSystemMessage = "You are a helpful assistant."


@prompt
def PointwiseLLMJudgePrompt(question: str, prediction: str) -> str:
    """
    [Instruction]
    Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".

    [Question]
    {{question}}

    [The Start of Assistant's Answer]
    {{prediction}}
    [The End of Assistant's Answer]
    """


backend_obj, prompt_obj = None, None


def llmjudge_fn(
    qrels, res, backend_type: str, model_name: str, rel=3, agg="max"
) -> int:
    """
    LLMasJudge function to evaluate the prediction against the gold standard.
    """

    pta.validate.columns(qrels, includes=["query_id", "relevance", "text"])
    pta.validate.columns(res, includes=["query_id", "qanswer"])

    assert len(res), "Empty res df provided"
    assert len(qrels), "Empty qrels df provided"
    qrels = qrels[qrels.relevance >= rel]
    assert len(qrels), "No qrels found with minimum label of %d" % rel

    global backend_obj
    global prompt_obj
    if backend_obj is None:
        backend_obj = get_backend(backend_type, model_name)
        prompt_obj = PromptTransformer(
            PointwiseLLMJudgePrompt,
            backend_obj.model_name_or_path,
            system_message=PointwiseLLMJudgeSystemMessage,
        )

    predictions = res["qanswer"].to_list()
    assert len(predictions) == 1, "Unexpected number of predictions"
    references = qrels["text"].to_list()
    # duplicate the prediction for the nbr of ground truths
    predictions = predictions * len(references)

    # create the prompt
    prompts = [
        prompt_obj.create_prompt(
            {
                "prediction": p,
                "reference": r,
            }
        )
        for p, r in zip(predictions, references)
    ]

    outputs = backend_obj.generate(prompts)
    parsed_ints = []
    for output in outputs:
        parsed_ints.append(int(output.split()[0]))
    assert len(parsed_ints) == len(outputs), "Unexpected number of parsed integers"
    assert len(parsed_ints) == len(predictions), "Unexpected number of parsed integers"
    # aggregate the scores
    if agg == "max":
        return max(parsed_ints)
    elif agg == "avg":
        return sum(parsed_ints) / len(parsed_ints)
    elif agg == "sum":
        return sum(parsed_ints)
    elif agg == "min":
        return min(parsed_ints)
    elif agg == "none":
        return parsed_ints
    else:
        raise ValueError(f"Unknown aggregation method: {agg}")


def LLMasJudge(backend_type, model_name_or_path):
    return ir_measures.define_byquery(
        lambda qrels, res: llmjudge_fn(
            qrels, res, backend_type=backend_type, model_name=model_name_or_path
        ),
        name="LLMasJudge",
        support_cutoff=False,
    )


__all__ = ["LLMasJudge"]
