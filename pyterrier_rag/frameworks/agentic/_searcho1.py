from ._base import AgenticRAG
from typing import Literal, Union
import pyterrier as pt
import pyterrier_alpha as pta
import pandas as pd

def get_singleqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Who got the first Nobel Prize in Physics?\"\n"
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
        "Question: \"Alice David is the voice of Lara Croft in a video game developed by which company?\"\n"
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

    def __init__(self, 
            retriever,
            backend,
            top_k=8,
            max_turn=6,
            analyse_results : bool = False,
            prompt_type : Union[Literal['single'], Literal['multi']]='single',
            **kwargs):
        super().__init__(
            retriever,
            backend,
            top_k=top_k,
            max_turn=max_turn,
            prompt_template=self._get_prompt_template(prompt_type, max_turn, backend.model_id),
            **kwargs,
        )
        self.start_search_tag = "<|begin_search_query|>"
        self.end_search_tag = "<|end_search_query|>"
        self.start_results_tag = "<|begin_of_documents|>"
        self.end_results_tag = "<|end_of_documents|>"

        # TODO: plug in analyser() as a replacement of super.format_docs(). I think
        # format_docs() could be a transformer
        assert not analyse_results, "analyse_results not yet implemented"

    def _get_prompt_template(self, prompt_type: str, max_turns, model_name):
        rtr = None
        if prompt_type == 'single':
            rtr = get_singleqa_search_o1_instruction(max_turns)
        elif prompt_type == 'multi':
            rtr = get_multiqa_search_o1_instruction(max_turns)

        if 'qwq' in model_name.lower():
            # qwq models are tuned for thinking step by step
            rtr += (
                'Please answer the following question. '
                'You should provide your final answer in the format <answer>YOUR ANSWER</answer>.\n\n' #'You should provide your final answer in the format \\boxed\{YOUR_ANSWER}.\n\n'
                'Question:\n{question}\n\n'
            )
        else:
            rtr += (
                'Please answer the following question. You should think step by step to solve it.\n\n'
                'You should provide your final answer in the format <answer>YOUR ANSWER</answer>.\n\n' #'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
                'Question:\n{question}\n\n'
            )

        return rtr


def analyser(backend):
    def _analyser(df_res):
        pta.validate.results_frame(includes=['query', 'docno', 'text'])
        prompt = ANALYSIS_PROMPT.format(search_query=df_res.iloc[0]['query'],
                                        prev_reasoning=df_res.iloc[0]['output'],
                                        documents="\n\n".join(df_res['text'].tolist()))
        return pd.DataFrame([df_res.iloc[0]['qid'], prompt], columns=['qid', 'output'])
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
