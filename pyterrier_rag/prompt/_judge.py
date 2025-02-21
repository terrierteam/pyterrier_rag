from outlines import prompt

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
