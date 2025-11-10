import re
from typing import Literal

import torch
import pyterrier as pt
import pyterrier_alpha as pta
import pandas as pd

from ._base import AgenticRAG
from ... import VLLMBackend, HuggingFaceBackend, Backend


def get_singleqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        'Question: "Who got the first Nobel Prize in Physics?"\n'
        "Assistant thinking steps:\n"
        "- I need to find out who was awarded the first Nobel Prize in Physics.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>first Nobel Prize in Physics winner<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )


def get_multiqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        'Question: "Alice David is the voice of Lara Croft in a video game developed by which company?"\n'
        "Assistant thinking steps:\n"
        "- I need to find out who voices Lara Croft in the video game.\n"
        "- Then, I need to determine which company developed that video game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Alice David Lara Croft voice<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant thinks: The search results indicate that Alice David is the voice of Lara Croft in a specific video game. Now, I need to find out which company developed that game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>video game developed by Alice David Lara Croft<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )


class SearchO1(AgenticRAG):

    def __init__(
        self,
        retriever: pt.Transformer,
        backend: Backend,
        top_k: int = 10,
        max_turn: int = 10,
        prompt_type: Literal["single", "multi"] = "single",
        **kwargs,
    ):
        super().__init__(
            retriever,
            backend,
            top_k=top_k,
            max_turn=max_turn,
            prompt_template=self._get_prompt_template(prompt_type, max_turn, backend.model_id),
            start_search_tag="<|begin_search_query|>",
            end_search_tag="<|end_search_query|>",
            start_results_tag="<|begin_of_documents|>",
            end_results_tag="<|end_of_documents|>",
            start_answer_tag="<answer>",
            end_answer_tag="</answer>",
            **kwargs,
        )

    @staticmethod
    def from_vllm(
        retriever: pt.Transformer,
        model: str,
        backend_args: dict | None = None,
        **kwargs,
    ) -> "SearchO1":
        if not backend_args:
            backend_args = {
                "model_args": {
                    "gpu_memory_utilization": 0.8,
                    "dtype": "bfloat16",
                    "max_model_len": 16384,
                },
                "generation_args": {
                    "temperature": 0.7,
                    "max_tokens": 2048,
                    "top_p": 0.8,
                    "top_k": 20,
                },
            }

        backend = VLLMBackend(model, **backend_args)
        return SearchO1(retriever, backend=backend, **kwargs)

    @staticmethod
    def from_hf(
        retriever: pt.Transformer,
        model: str,
        backend_args: dict | None = None,
        **kwargs,
    ) -> "SearchO1":
        if not backend_args:
            backend_args = {
                "model_args": {
                    "torch_dtype": torch.bfloat16,
                    "device_map": "auto",
                },
                "generation_args": {
                    "max_new_tokens": 2048,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                },
            }

        backend = HuggingFaceBackend(model, **backend_args)
        backend.tokenizer.padding_side = "left"  # for decoder-only models
        # to suppress warning
        backend._model.generation_config.pad_token_id = backend.tokenizer.pad_token_id
        return SearchO1(retriever, backend=backend, **kwargs)

    def wrap_search_results(self, docs: pd.DataFrame, state: dict, **kwargs) -> str:
        if not isinstance(docs, pd.DataFrame) or docs.empty:
            return f"\n\n{self.start_results_tag}\n{self.end_results_tag}\n\n"

        analysis = self.extract_answer(self._analyse(docs, state), mode="infogen")
        text = f"\n\n{self.start_results_tag}\n{analysis}\n{self.end_results_tag}\n\n"
        state["output"] += text

        return text

    def _analyse(self, docs: pd.DataFrame, state: dict) -> str:
        """Reduce documents for one single query."""

        prompt = ANALYSIS_PROMPT.format(
            search_query=docs.iloc[0]["query"],
            prev_reasoning=state.get("output", ""),
            documents="\n\n".join(self._format_retrieval_docs(docs)),
        )

        # TODO: support batch generation; adapt the main loop
        return self.backend.generate([prompt])[0].text

    def _format_retrieval_docs(self, docs: pd.DataFrame) -> list[str]:
        """Formatting code adapted from upstream."""

        def truncate_text(text, max_words=360):
            words = text.split()
            if len(words) <= max_words:
                return text
            else:
                return " ".join(words[:max_words])

        _docs = []
        for row in docs.itertuples():
            text = truncate_text(row.text)
            title = getattr(row, "title", None)
            if isinstance(title, str) and title:
                title = title.strip('"')
                text = text.removeprefix(title).lstrip()
                _docs.append(f"Title: {title}\nText: {text}")
            else:
                _docs.append(f"Text: {text}")

        return _docs

    def _get_prompt_template(self, prompt_type: str, max_turns: int, model_name: str) -> str:
        rtr = None
        if prompt_type == "single":
            rtr = get_singleqa_search_o1_instruction(max_turns)
        elif prompt_type == "multi":
            rtr = get_multiqa_search_o1_instruction(max_turns)

        if "qwq" in model_name.lower():
            # qwq models are tuned for thinking step by step
            rtr += (
                "Please answer the following question. "
                "You should provide your final answer in the format <answer>YOUR ANSWER</answer>.\n\n"  #'You should provide your final answer in the format \\boxed\{YOUR_ANSWER}.\n\n'
                "Question:\n{question}\n\n"
            )
        else:
            rtr += (
                "Please answer the following question. You should think step by step to solve it.\n\n"
                "You should provide your final answer in the format <answer>YOUR ANSWER</answer>.\n\n"  #'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
                "Question:\n{question}\n\n"
            )

        return rtr

    @staticmethod
    def extract_answer(output, mode="gen"):
        extracted_text = ""
        if mode == "codegen":
            # Extract the code between ```python and ```
            pattern = r"```python\s*(.*?)\s*```"
            matches = re.findall(pattern, output, re.DOTALL | re.IGNORECASE)
            if matches:
                extracted_text = matches[-1].strip()  # Take the last match
        elif mode == "infogen":
            # Extract content after **Final Information** or **Modified Reasoning Steps**
            pattern_info = "\n**Final Information**"
            pattern_step = "\n**Modified Reasoning Steps**"
            if pattern_info in output:
                extracted_text = output.split(pattern_info)[-1].replace("\n", "").strip("```").strip()
            elif pattern_step in output:
                extracted_text = output.split(pattern_step)[-1].strip("```").strip()
            else:
                extracted_text = "No helpful information found."
        else:
            # Existing extraction logic for 'gen' and 'choose' modes
            pattern = r"\\boxed\{(.*)\}"
            matches = re.findall(pattern, output)
            if matches:
                extracted_text = matches[-1]  # Take the last match
                if mode in ["choose", "qa"]:
                    # Handle 'choose' mode
                    inner_pattern = r"\\text\{(.*)\}"
                    inner_matches = re.findall(inner_pattern, extracted_text)
                    if inner_matches:
                        extracted_text = inner_matches[-1]  # Take the last match
                    extracted_text = extracted_text.strip("()")
        return extracted_text


def analyser(backend):
    def _analyser(df_res):
        pta.validate.results_frame(includes=["query", "docno", "text"])
        prompt = ANALYSIS_PROMPT.format(
            search_query=df_res.iloc[0]["query"],
            prev_reasoning=df_res.iloc[0]["output"],
            documents="\n\n".join(df_res["text"].tolist()),
        )
        return pd.DataFrame([df_res.iloc[0]["qid"], prompt], columns=["qid", "output"])

    return pt.apply.by_query(_analyser) >> backend.text_generator()


ANALYSIS_PROMPT = """**Task Instruction:**
You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. 
Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **Extract Relevant Information:**
- Select the information from the Searched Web Pages that directly contributes to advancing the **Previous Reasoning Steps**.
- Ensure that the extracted information is accurate and relevant.

3. **Output Format:**
- **If the web pages provide helpful information for current search query:** Present the information beginning with `**Final Information**` as shown below.
**Final Information**

[Helpful information]

- **If the web pages do not provide any helpful information for current search query:** Output the following text.

**Final Information**

No helpful information found.

**Inputs:**
- **Previous Reasoning Steps:**  
{prev_reasoning}

- **Current Search Query:**  
{search_query}

- **Searched Web Pages:**  
{documents}

Now you should analyze each web page and find helpful information based on the current search query "{search_query}" and previous reasoning steps.
"""
