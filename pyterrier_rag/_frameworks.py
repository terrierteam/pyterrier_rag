from typing import Iterable, Optional

from outlines import prompt
import pyterrier as pt

from pyterrier_rag.prompt import PromptTransformer, PromptConfig, ContextConfig

"""
Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions (IRCOT) ACL 2023

Paper: https://arxiv.org/abs/2212.10509
Implementation Derived from: https://github.com/RUC-NLPIR/FlashRAG/blob/main/flashrag/pipeline/active_pipeline.py#L925
"""

ircot_system_message = 'You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. This task is illustrated through demonstrations, each consisting of a document set paired with a relevant question and its multi-hop reasoning thoughts. Your task is to generate one thought for current step, DON\'T generate the whole thoughts at once! If you reach what you believe to be the final step, start with "So the answer is:".'


@prompt
def ircot_prompt(context: str, query: str) -> str:
    """{context}Question: {query}\nThought:"""


@prompt
def ircot_example_format(text: str, title: str = None) -> str:
    """
    {% if title != None %}
    Wikipedia Title: {{title}}
    {% endif %}
    {{text}}
    """


class IRCOT(pt.Transformer):
    def __init__(
        self,
        retriever: pt.Transformer,
        reader: pt.Transformer,
        input_field: str = "query",
        output_field: str = "qanswer",
        prompt: Optional[pt.Transformer] = None,
        prompt_config: Optional[PromptConfig] = None,
        context_config: Optional[ContextConfig] = None,
        max_docs: int = 10,
        max_iterations: int = -1,
        exit_condition: callable = lambda x: "so the answer is"
        in x["qanswer"].iloc[0].lower(),
    ):
        self.retriever = retriever % max_docs
        self.reader = reader
        self.input_field = input_field
        self.output_field = output_field
        self.exit_condition = exit_condition
        self.prompt_config = prompt_config
        self.context_config = context_config
        self.prompt = prompt

        self.max_docs = max_docs
        self.max_iterations = max_iterations

        self.__post_init__()

    def __post_init__(self):
        if self.prompt_config is None:
            self.prompt_config = self._make_default_prompt_config()
        if self.context_config is None:
            self.context_config = self._make_default_context_config()
        if self.prompt is None:
            self.prompt = PromptTransformer(
                config=self.prompt_config, context_config=self.context_config
            )

    def _exceeded_max_iterations(self, iter):
        return self.max_iterations > 0 and iter >= self.max_iterations

    def _make_default_prompt_config(self):
        return PromptConfig(
            model_name_or_path=self.reader.model_name_or_path,
            system_message=ircot_system_message,
            instruction=ircot_prompt,
            output_field="qanswer",
            input_fields=["query", "context"],
        )

    def _make_default_context_config(self):
        return ContextConfig(
            in_fields=["text"],
            out_field="context",
            tokenizer=self.reader.tokenizer,
            max_length=self.reader.max_input_length,
            max_elements=self.max_docs,
            intermediate_format=ircot_example_format,
        )

    def transform_by_query(self, inp: Iterable[dict]) -> Iterable[dict]:
        inp = list(inp)
        qid = inp[0]["qid"]
        query = inp[0]["query"]
        for row in inp:
            assert (
                row["query"] == query
            ), "All rows must have the same query for `transform_by_query`"

        prev = []
        top_k_docs = self.retriever.search(query)
        iter = 1
        stop = False

        while not stop:
            prompt = self.prompt.transform(top_k_docs)
            output = self.reader.transform(prompt)

            if self.exit_condition(output) or self._exceeded_max_iterations(iter):
                stop = True
                break
            else:
                prev.append(output[self.output_field].iloc[0])
                top_k_docs.append(
                    self.retriever.search(output[self.output_field].iloc[0])
                )
                top_k_docs.sort_values(by="score", ascending=False, inplace=True)
                top_k_docs.drop_duplicates(subset=["docid"], inplace=True)
                top_k_docs = top_k_docs.head(self.max_docs)
                iter += 1

        qanswer = output[self.output_field].iloc[0]
        return [{"qid": qid, "query": query, "qanswer": qanswer}]
