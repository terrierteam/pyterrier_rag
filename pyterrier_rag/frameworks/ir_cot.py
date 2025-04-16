from typing import Iterable, Optional

from outlines import prompt
import pyterrier as pt
import pyterrier_alpha as pta

from pyterrier_rag.backend import Backend
from pyterrier_rag.readers import Reader
from pyterrier_rag.prompt import PromptTransformer, PromptConfig, ContextConfig

"""
Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions (IRCOT) ACL 2023

Paper: https://arxiv.org/abs/2212.10509
Implementation Derived from: https://github.com/RUC-NLPIR/FlashRAG/blob/main/flashrag/pipeline/active_pipeline.py#L925
"""

ircot_system_message = 'You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. This task is illustrated through demonstrations, each consisting of a document set paired with a relevant question and its multi-hop reasoning thoughts. Your task is to generate one thought for current step, DON\'T generate the whole thoughts at once! If you reach what you believe to be the final step, start with "So the answer is:".\n\n Wikipedia Title: Kurram Garhi\nKurram Garhi is a small village located near the city of Bannu, which is the part of Khyber Pakhtunkhwa province of Pakistan. Its population is approximately 35000. Barren hills are near this village. This village is on the border of Kurram Agency. Other nearby villages are Peppal, Surwangi and Amandi Kala.\n\nWikipedia Title: 2001–02 UEFA Champions League second group stage\nEight winners and eight runners- up from the first group stage were drawn into four groups of four teams, each containing two group winners and two runners- up. Teams from the same country or from the same first round group could not be drawn together. The top two teams in each group advanced to the quarter- finals.\n\nWikipedia Title: Satellite tournament\nA satellite tournament is either a minor tournament or event on a competitive sporting tour or one of a group of such tournaments that form a series played in the same country or region.\n\nWikipedia Title: Trojkrsti\nTrojkrsti is a village in Municipality of Prilep, Republic of Macedonia.\n\nWikipedia Title: Telephone numbers in Ascension Island\nCountry Code:+ 247< br> International Call Prefix: 00 Ascension Island does not share the same country code( +290) with the rest of St Helena.\n\nQuestion: Are both Kurram Garhi and Trojkrsti located in the same country?\nThought: Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country. So the answer is: no.\n\n'


@prompt
def ircot_prompt(context: str, query: str, prev: str) -> str:
    """{context}Question: {query}\nThought:\n\n{prev}"""


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
        backend: Backend,
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
        self.backend = backend
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
        self.reader = Reader(backend=self.backend, prompt=self.prompt)

    def _exceeded_max_iterations(self, iter):
        return self.max_iterations > 0 and iter >= self.max_iterations

    def _make_default_prompt_config(self):
        return PromptConfig(
            model_name_or_path=self.backend.model_name_or_path,
            system_message=ircot_system_message,
            instruction=ircot_prompt,
            output_field="qanswer",
            input_fields=["query", "context", "prev"],
        )

    def _make_default_context_config(self):
        return ContextConfig(
            in_fields=["text"],
            out_field="context",
            tokenizer=self.backend.tokenizer,
            max_length=self.backend.max_input_length,
            max_elements=self.max_docs,
            intermediate_format=ircot_example_format,
        )

    @pta.transform.by_query(add_ranks=False)
    def transform_iter(self, inp: Iterable[dict]) -> Iterable[dict]:
        return self.transform_by_query(inp)

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
        top_k_docs["prev"] = ""
        iter = 1
        stop = False

        while not stop:
            output = self.reader.transform(top_k_docs)

            if self.exit_condition(output) or self._exceeded_max_iterations(iter):
                stop = True
                break
            else:
                prev.append(output[self.output_field].iloc[0])
                top_k_docs.append(
                    self.retriever.search(output[self.output_field].iloc[0])
                )
                top_k_docs.sort_values(by="score", ascending=False, inplace=True)
                top_k_docs.drop_duplicates(subset=["docno"], inplace=True)
                top_k_docs = top_k_docs.head(self.max_docs)
                top_k_docs["prev"] = "\n\n".join(prev)
                iter += 1

        qanswer = output[self.output_field].iloc[0]
        return [{"qid": qid, "query": query, "qanswer": qanswer}]


__all__ = ["IRCOT"]
