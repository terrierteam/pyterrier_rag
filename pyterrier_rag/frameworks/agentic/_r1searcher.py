from ._base import AgenticRAG
from typing import Literal

class R1Searcher(AgenticRAG):

    DEFAULT_MODEL = "XXsongLALA/Qwen-2.5-7B-base-RAG-RL"

    @staticmethod 
    def from_vllm(*args, model=DEFAULT_MODEL, **kwargs):
        from ... import VLLMBackend
        backend = VLLMBackend.from_dsn(f"vllm:{model}")
        return R1Searcher(*args, backend=backend, **kwargs)

    @staticmethod 
    def from_hf(*args, model=DEFAULT_MODEL, **kwargs):
        from ... import HuggingFaceBackend
        backend = HuggingFaceBackend.from_dsn(f"huggingface:{model}")
        return R1Searcher(*args, backend=backend, **kwargs)

    """
    prompt_type Description:
     - v0: Single-step reasoning. When encountering uncertain knowledge, use <|begin_of_query|>keyword<|end_of_query|> to search. Suitable for general Q&A.
     - v1: Multi-step/sub-question reasoning. The question is split into sub-questions, each of which is searched using <|begin_of_query|>kw1\tkw2<|end_of_query|>. Keywords are separated by tabs. Suitable for multi-hop/complex Q&A.
     - v2: Multi-step reasoning + keyword search. The search query only allows keyword lists (separated by \t), not complete sentences. Suitable for keyword-only search scenarios.
     - v3: Judgment reasoning. Designed for yes/no questions. The answer after reasoning must be yes or no. The search method is the same as v0. Suitable for judgment-type Q&A.
    
    """
    def __init__(self, 
             retriever,
             backend,
             top_k=8,
             max_turn=6,
             prompt_type : Literal['v1', 'v2', 'v3']='v1',
             **kwargs):
        super().__init__(
            retriever,
            backend,
            top_k=top_k,
            max_turn=max_turn,
            prompt=self.get_prompt(prompt_type),
            **kwargs,
        )

        self.start_search_tag = "<|begin_of_query|>"
        self.end_search_tag = "<|end_of_query|>"
        self.start_results_tag = "<|begin_of_documents|>"
        self.end_results_tag = "<|end_of_documents|>"

    def get_prompt(self, prompt_type: str):
        if prompt_type == 'v0':
            return V0_PROMPT
        elif prompt_type == 'v1':
            return V1_PROMPT
        elif prompt_type == 'v2':
            return V2_PROMPT
        elif prompt_type == 'v3':
            return V3_PROMPT


V0_PROMPT = """The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the thinking process, the Assistant can perform searching for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only keywords) here <|end_of_query|>". **A query must involve only a single triple**.
Then, the system will provide the Assistant with helpful information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".\n\nUser:{question}\nAssistant: <think>"""

V1_PROMPT = """The User asks a question, and the Assistant solves it.
Use these tags ONLY: <think>...</think>, <|begin_of_query|>...<|end_of_query|>, <|begin_of_documents|>...<|end_of_documents|>, <answer>...</answer>.
General protocol:
1) Inside <think>, decompose the question if needed and decide what information is missing.
2) When external knowledge is needed, output EXACTLY one line:
   <|begin_of_query|> keyword1\tkeyword2\t... <|end_of_query|>
   - Include the core entity/subject and the essential property/constraint keywords.
   - Add common aliases/synonyms (English and/or Chinese) when helpful.
   - Immediately STOP after <|end_of_query|>. Do NOT output anything else until <|begin_of_documents|> is provided.
3) After I return <|begin_of_documents|> ... <|end_of_documents|>, resume <think> to extract the needed facts:
   - Prefer explicit statements that directly support the requirement.
   - If evidence is insufficient or off-topic, refine keywords and SEARCH again.
4) Only when there is clear supporting evidence in <|begin_of_documents|> ... <|end_of_documents|>, output:
   <answer> final answer here </answer>

Output rules:
- Keep <think> concise; do not reveal chain-of-thought beyond the tag.
- Do NOT output <answer> until evidence from <|begin_of_documents|> is found.
- If still uncertain after several searches, continue searching; do not guess.
- Do NOT output <answer> until a clear supporting statement is found in <|begin_of_documents|>.
- If the retrieved information does not directly answer the question, refine the keywords and <|begin_of_query|>...<|end_of_query|> again.
User:{question}
Assistant: <think>
"""

V2_PROMPT = """The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the thinking process, **the Assistant can perform searching** for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only list keywords separated by "\t" instead of the complete sentence , such as **"keyword_1 \t keyword_2 \t..."**)<|end_of_query|>". **A query must involve only a single triple**.
Then, the search system will provide the Assistant with the retrieval information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".

User:{question}
Assistant: <think>"""

V3_PROMPT = """The User asks a **Judgment question**, and the Assistant solves it. 
The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer. 
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here (yes or no) </answer>". 
During the thinking process, the Assistant can perform searching for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only keywords) here <|end_of_query|>". 
Then, the system will provide the Assistant with helpful information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>". 
The final answer **must be yes or no**.\n\nUser:{question}\nAssistant: <think>"""
