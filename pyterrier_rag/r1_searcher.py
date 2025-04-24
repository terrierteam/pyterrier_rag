import pyterrier as pt
import pyterrier_alpha as pta
from transformers import AutoTokenizer
from typing import Dict, Any

def process_text(examples,tokenizer,type=None):

    base_prompt_v0 = """The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the thinking process, the Assistant can perform searching for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only keywords) here <|end_of_query|>". **A query must involve only a single triple**.
Then, the system will provide the Assistant with helpful information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".\n\nUser:{question}\nAssistant: <think>"""

    base_prompt_v1 = """The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the reasoning process, the Assistant will break down the original question into sub-questions and address them step by step.
For each sub-question, **the Assistant can perform searching** for uncertain knowledge using the format: "<|begin_of_query|> keyword1\tkeyword2\t... <|end_of_query|>".
**The query must consist of straightforward and essential keywords separated by "\t"**. Furthermore, **the query must involve only a single triple to address a sub-question**.
Then, the search system will provide the Assistant with relevant information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".

User:{question}
Assistant: <think>"""
    base_prompt_v2="""The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the thinking process, **the Assistant can perform searching** for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only list keywords separated by "\t" instead of the complete sentence , such as **"keyword_1 \t keyword_2 \t..."**)<|end_of_query|>". **A query must involve only a single triple**.
Then, the search system will provide the Assistant with the retrieval information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".

User:{question}
Assistant: <think>"""

    base_prompt_v3 = """The User asks a **Judgment question**, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer. The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here (yes or no) </answer>". During the thinking process, the Assistant can perform searching for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only keywords) here <|end_of_query|>". Then, the system will provide the Assistant with helpful information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>". The final answer **must be yes or no**.\n\nUser:{question}\nAssistant: <think>"""

    if type == "v0":
        question = examples["question"]
        prompt = base_prompt_v0.format(question=question)
        examples["chat_prompt"] = prompt
    elif type=="v1":
        question = examples["question"]
        prompt = base_prompt_v1.format(question=question)
        examples["chat_prompt"] = prompt
    elif type=="v2":
        question = examples["question"]
        prompt = base_prompt_v2.format(question=question)
        examples["chat_prompt"] = prompt
    elif type=="v3":
        question = examples["question"]
        prompt = base_prompt_v3.format(question=question)
        examples["chat_prompt"] = prompt
    else:
        raise ValueError("unknown type")
    return examples

class R1Searcher(pt.Transformer):
    """
    Another Agentic RAG model.

    Code framework comes from https://github.com/RUCAIBox/R1-Searcher/blob/main/evaluation/eval_search_loacl.py, but it simplified to avoid
    parallel processing.

    Input columns:
     - qid
     - query
    
    Output columns:
     - qid
     - query
     - output (output of the model)
     - qanswer (extracted from the output)
     - iteration (how many thought iterations)
     - all_queries (what was sent to the retriever)
     - stop_reason_final (why did the generation stop?): "finished", "many_retrieve", "query_inst_error", "shot_down"

    R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning
    Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, Ji-Rong Wen
    https://arxiv.org/abs/2503.05592
    """

    def __init__(self, 
                 retriever : pt.Transformer, 
                 model_path : str = "XXsongLALA/Qwen-2.5-7B-base-RAG-RL", 
                 model_kw_args : Dict[str,Any] = {'tensor_parallel_size' : 1, 'gpu_memory_utilization' : 0.95},
                 temp : float = 0, 
                 top_k : int = 5, 
                 verbose : bool = False,
                 prompt_type : str ='v3'):
        # delay importing vllm until needed
        from vllm import LLM, SamplingParams
        self.retriever = retriever
        stop_tokens = ["<|im_end|>", "<|endoftext|>", "<|end_of_query|>", "</answer>"]
        self.sampling_params = SamplingParams(temperature=temp, top_p=0.95, max_tokens=512, stop=stop_tokens)
        if model_path is not None:
            self.llm = LLM(model=model_path, trust_remote_code=True, **model_kw_args)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_iterations = 16
        self.top_k = top_k
        self.prompt_type = prompt_type
        self.verbose = verbose

    def clone_for_retriever(self, new_retriever) -> 'R1Searcher':
        """
        Make a copy of this model with a new retiever. This ensures that the model doesnt need to be loaded multiple times for 
        experiments that vary the retriever 
        """
        rtr = R1Searcher(new_retriever, model_id = None)
        rtr.model = self.model
        rtr.tokenizer = self.tokenizer
        rtr.max_iterations = self.max_iterations
        rtr.top_k = self.top_k
        rtr.prompt_type = self.prompt_type
        rtr.verbose = self.verbose
        return rtr

    @pta.transform.by_query(add_ranks=False)
    def transform_iter(self, inp):
        inp = next(inp)
        question = inp["query"]
        qid = inp["qid"]
        
        ds = process_text({'question' : question}, self.tokenizer, type=self.prompt_type)
        continued_answer = dict(ds) #copy.deepcopy?
        continued_answer["gen_text_store"] = ""
        queries = []

        for k in range(self.max_iterations):
            output = self.llm.generate([continued_answer['chat_prompt']], self.sampling_params, use_tqdm = self.verbose)[0]
            prompt = output.prompt
            gen_text_store = continued_answer["gen_text_store"]
            stop_reason = output.outputs[0].stop_reason
            generated_text = output.outputs[0].text
            
            if k == 9: # original code breaks out here. 
                original_data = {
                        "qid" : qid,
                        "query":question,
                        "all_queries" : queries,
                        'iteration' : k,
                        "output":generated_text,
                        "stop_reason_final": "many_retrieve",
                        "qanswer": "I don't know."
                }

                return [original_data]

            if "<answer>" in generated_text and stop_reason=="</answer>":
                original_data = {
                    "qid" : qid,
                    "query":question,
                    'iteration' : k,
                    "all_queries" : queries,
                    "qanswer": generated_text.split("<answer>")[-1].split("</answer>")[0],
                    "stop_reason_final": "finished",
                    "output": gen_text_store + generated_text + "</answer>",
                }
                return [original_data]
        
            elif "<|begin_of_query|>" in generated_text and stop_reason=="<|end_of_query|>":
                query = generated_text.split("<|begin_of_query|>")[-1].split("<|end_of_query|>")[0]
                query = query.replace('"',"").replace("'","").replace("\t"," ").replace("...","")
                if query:
                        queries.append((k, query))
                        original_data = {
                            "chat_prompt":prompt + generated_text.strip(), #+ "<|end_of_query|> "+ "\n\nThe retrieved content are:\n<tool_call>\n"  +  doc_content + "\n</tool_call>\n\n",
                            "stop_reason": stop_reason,
                            "gen_text_store": gen_text_store + generated_text.strip() #+ "<|end_of_query|> "+ "\n\nThe retrieved content are:\n<tool_call>\n"  +  doc_content + "\n</tool_call>\n\n",
                            }
                        
                        results = (self.retriever % self.top_k).search(query, qid="%s-%d" % (qid, k))
                        if len(results) > 0:                
                            doc_content_list = [f"({j+1}){doc_content}\n" for j, doc_content in enumerate(results["text"])]
                            doc_content = ''.join(doc_content_list)
                        else:
                            doc_content = 'None'
                        continued_text_now = original_data
                        continued_text_now["chat_prompt"] = continued_text_now["chat_prompt"] + "<|end_of_query|>\n\n"+ "<|begin_of_documents|>\n" +  doc_content + "<|end_of_documents|>\n\n"
                        continued_text_now["gen_text_store"] = continued_text_now["gen_text_store"] + "<|end_of_query|>\n\n"+ "<|begin_of_documents|>\n" +  doc_content + "<|end_of_documents|>\n\n"
                        
                        continued_answer = continued_text_now
                        continue
                else: # we saw begin_of_query and end_of_query, but no valid query found.
                    original_data = {
                        "qid" : qid,
                        "query":question,
                        "output": gen_text_store + generated_text.strip(),
                        'iteration' : k,
                        "all_queries" : queries,
                        "stop_reason_final": "query_inst_error",
                        "qanswer": "I don't know."
                    }
                    return [original_data]
            else:
                original_data = {
                    "qid" : qid,
                    "query": question,
                    'iteration' : k,
                    "all_queries" : queries,
                    'output' : gen_text_store + generated_text.strip(),
                    "stop_reason_final": "shot_down",
                    "qanswer": "I don't know."
                }
                return [original_data]
        