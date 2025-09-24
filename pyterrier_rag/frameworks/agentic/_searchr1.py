from ._base import AgenticRAG
from ... import Backend, HuggingFaceBackend

class SearchR1(AgenticRAG):

    DEFAULT_MODEL = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo"

    @staticmethod 
    def from_vllm(*args, model=DEFAULT_MODEL, **kwargs):
        from ... import VLLMBackend
        backend = VLLMBackend(model)
        return SearchR1(*args, backend=backend, **kwargs)
    
    @staticmethod 
    def from_hf(*args, model=DEFAULT_MODEL, **kwargs):
        from ... import HuggingFaceBackend
        backend = HuggingFaceBackend(model)
        return SearchR1(*args, backend, **kwargs)

    def __init__(self,
        retriever,
        backend,
        temperature = 0.7,
        top_k = 8,
        max_turn = 10,
        max_tokens = 512,
        **kwargs
    ):
        
        super().__init__(
            retriever,
            backend,
            prompt = self._get_prompt(),
            top_k = top_k,
            max_turn = max_turn,
            max_tokens = max_tokens,
            temperature = temperature,
            start_search_tag="<search>",
            end_search_tag="</search>",
            start_results_tag="<information>",
            end_results_tag="</information>",
            **kwargs
        )
       
       
    def _get_prompt(self):
        prompt = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: What is the capital of China?
User:{question}
Assistant: <think>"""
        return prompt

    