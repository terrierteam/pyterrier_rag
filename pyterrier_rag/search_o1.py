# Modified from https://github.com/sunnynexus/Search-o1
# Changes made by Jinyuan on 2025-04-14

import re 
import torch 
from typing import List, Dict, Iterable

import pyterrier as pt
import pyterrier_alpha as pta
from pyterrier_rag.backend import HuggingFaceBackend, StopWordCriteria

# Define special tokens
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"


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

def get_task_instruction_openqa(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Please answer the following question. '
            'You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'Please answer the following question. You should think step by step to solve it.\n\n'
            'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    return user_prompt

def get_webpage_to_reasonchain_instruction(prev_reasoning, search_query, document):
    return f"""**Task Instruction:**

You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

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
{document}

Now you should analyze each web page and find helpful information based on the current search query "{search_query}" and previous reasoning steps.
"""

def extract_answer(output, mode='gen'):
    extracted_text = ''
    if mode == 'codegen':
        # Extract the code between ```python and ```
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, output, re.DOTALL | re.IGNORECASE)
        if matches:
            extracted_text = matches[-1].strip()  # Take the last match
    elif mode == 'infogen':
        # Extract content after **Final Information** or **Modified Reasoning Steps**
        pattern_info = "\n**Final Information**"
        pattern_step = "\n**Modified Reasoning Steps**"
        if pattern_info in output:
            extracted_text = output.split(pattern_info)[-1].replace("\n","").strip("```").strip()
        elif pattern_step in output:
            extracted_text = output.split(pattern_step)[-1].strip("```").strip()
        else:
            extracted_text = "No helpful information found."
    else:
        # Existing extraction logic for 'gen' and 'choose' modes
        pattern = r'\\boxed\{(.*)\}'
        matches = re.findall(pattern, output)
        if matches:
            extracted_text = matches[-1]  # Take the last match
            if mode in ['choose', 'qa']:
                # Handle 'choose' mode
                inner_pattern = r'\\text\{(.*)\}'
                inner_matches = re.findall(inner_pattern, extracted_text)
                if inner_matches:
                    extracted_text = inner_matches[-1]  # Take the last match
                extracted_text = extracted_text.strip("()")
    return extracted_text


def replace_recent_steps(origin_str, replace_str):

    """
    Replaces specific steps in the original reasoning steps with new steps.
    If a replacement step contains "DELETE THIS STEP", that step is removed.

    Parameters:
    - origin_str (str): The original reasoning steps.
    - replace_str (str): The steps to replace or delete.

    Returns:
    - str: The updated reasoning steps after applying replacements.
    """

    def parse_steps(text) -> Dict[int,str]:
        """
        Parses the reasoning steps from a given text.

        Parameters:
        - text (str): The text containing reasoning steps.

        Returns:
        - dict: A dictionary mapping step numbers to their content.
        """
        step_pattern = re.compile(r"Step\s+(\d+):\s*")
        steps = {}
        current_step_num = None
        current_content = []

        for line in text.splitlines():
            step_match = step_pattern.match(line)
            if step_match:
                # If there's an ongoing step, save its content
                if current_step_num is not None:
                    steps[current_step_num] = "\n".join(current_content).strip()
                current_step_num = int(step_match.group(1))
                content = line[step_match.end():].strip()
                current_content = [content] if content else []
            else:
                if current_step_num is not None:
                    current_content.append(line)
            
        # Save the last step if any
        if current_step_num is not None:
            steps[current_step_num] = "\n".join(current_content).strip()
            
        return steps

    # Parse the original and replacement steps
    origin_steps = parse_steps(origin_str)
    replace_steps = parse_steps(replace_str)

    # Apply replacements
    for step_num, content in replace_steps.items():
        if "DELETE THIS STEP" in content:
            # Remove the step if it exists
            if step_num in origin_steps:
                del origin_steps[step_num]
        else:
            # Replace or add the step
            origin_steps[step_num] = content

    # Sort the steps by step number
    sorted_steps = sorted(origin_steps.items())

    # Reconstruct the reasoning steps as a single string
    new_reasoning_steps = "\n\n".join([f"{content}" for num, content in sorted_steps])

    return new_reasoning_steps


class SearchO1(pt.Transformer):

    def __init__(
        self, 
        retriever : pt.Transformer, 
        generator: HuggingFaceBackend,
        max_turn: int=10, 
        max_retrieval_step: int=5,
        topk: int=10, 
        temperature: float=0.7, 
        top_p: float=0.8, 
        top_k: int=20,
        multihop_qa: bool=True, 
        **kwargs
    ):
        super().__init__()

        self.retriever = retriever
        self.generator = generator
        self.device = self.generator.device 
        self.tokenizer = self.generator.tokenizer
        self.max_turn = max_turn
        self.max_retrieval_step = max_retrieval_step
        self.topk = topk 
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.kwargs = kwargs

        self.multihop_qa = multihop_qa

    def get_init_prompt(self, question: str, multihop_qa: bool=False) -> str:

        if multihop_qa:
            instruction = get_multiqa_search_o1_instruction(self.max_retrieval_step)
        else:
            instruction = get_singleqa_search_o1_instruction(self.max_retrieval_step)
        
        if "qwq" in self.generator._model.config._name_or_path.lower():
            user_prompt = get_task_instruction_openqa(question, model_name="qwq")
        else:
            user_prompt = get_task_instruction_openqa(question)

        # try:
        #     prompt = [{"role": "user", "content": instruction + user_prompt}]
        #     prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        # except Exception:
        #     prompt = instruction + user_prompt 
        prompt = instruction + user_prompt 
        
        return prompt 
    
    def tokenizer_encode(self, prompts: List[str]) -> List[torch.Tensor]:
        inputs = self.tokenizer(prompts, padding=True, truncation=True, max_length=self.generator.text_max_length, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        return input_ids, attention_mask
    
    def generate(self, sequences: List[dict]) -> List[str]:

        prompts = [s['prompt'] for s in sequences]
        input_ids, attention_mask = self.tokenizer_encode(prompts)
        # generation with stop words 
        prompt_length = input_ids.shape[-1] 
        token_ids = self.generator._model.generate(
            input_ids, 
            attention_mask=attention_mask,
            do_sample=True, 
            temperature=self.temperature, 
            top_p=self.top_p, 
            top_k=self.top_k,
            max_new_tokens=self.generator.max_new_tokens,
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
            stopping_criteria=[StopWordCriteria(self.tokenizer, prompt_length, [END_SEARCH_QUERY, self.tokenizer.eos_token])]
        )
        generated_token_ids = token_ids[:, prompt_length:] 
        generated_texts = self.tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)
        return generated_texts
    
    def extract_search_query(self, text: str) -> str: 

        pattern = re.escape(BEGIN_SEARCH_QUERY) + r'(.*?)' + re.escape(END_SEARCH_QUERY)
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None 
    
    def retrieve_docs(self, search_queries: List[str]) -> List[List[dict]]:
        """
        retrieve documents from the retriever for a list of search queries.
        """
        retrieval_results = (self.retriever % self.topk)(pt.new.queries(search_queries))
        with pta.validate.any(retrieval_results) as v:
            # we need either passages with (text column), or sentences column
            v.result_frame(extra_columns=['text'], mode='passages')
            v.result_frame(extra_columns=['sentences'], mode='sentences')
        
        rtr = []    
        for qid, results in retrieval_results.groupby('qid'):
            rtr.append(results.to_dict('records'))
        
        return rtr
    
    def format_retrieval_docs(self, retrieval_results: List[List[dict]]) -> List[List[str]]:

        def truncate_text(text, max_words=360):
            words = text.split()
            if len(words) <= max_words:
                return text
            else:
                return " ".join(words[:max_words])

        retrieved_docs = [] 
        for retrieval_result in retrieval_results:
            docs = [] 
            for item in retrieval_result:
                title = item["title"] if "title" in item else None 
                text = item["text"] if "text" in item else " ".join(sent.strip() for sent in item["sentences"])
                text = truncate_text(text)
                if title:
                    docs.append(f"Title: {title}\nText: {text}")
                else:
                    docs.append(f"Text: {text}")
            retrieved_docs.append(docs)
        return retrieved_docs 
    
    def analyze_docs(self, sequences: List[dict], prev_reasonings: List[str], queries: List[str], retrieval_results: List[List[dict]]) -> List[str]: 
        """
        analyze the retrieved documents and return the analysis results. 
        """
        user_prompts = [] 
        retrieved_docs = self.format_retrieval_docs(retrieval_results)
        for prev_reasoning, query, docs in zip(prev_reasonings, queries, retrieved_docs):
            docs_info = "\n\n".join(docs)
            user_prompts.append(get_webpage_to_reasonchain_instruction(prev_reasoning, query, docs_info))
        
        prompts = user_prompts
        input_ids, attention_mask = self.tokenizer_encode(prompts)
        token_ids = self.generator._model.generate(
            input_ids, 
            attention_mask=attention_mask,
            do_sample=True, 
            temperature=self.temperature, 
            top_p=self.top_p, 
            top_k=self.top_k,
            max_new_tokens=self.generator.max_new_tokens,
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
        )
        generated_token_ids = token_ids[:, input_ids.shape[-1]:]  
        generated_texts = self.tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)

        extracted_infos = [extract_answer(text, mode="infogen") for text in generated_texts]

        outputs = [] 
        for prompt, text, info in zip(prompts, generated_texts, extracted_infos):
            outputs.append(
                {
                    "prompt": prompt, 
                    "raw_output": text, 
                    "extracted_info": info
                }
            )
        return outputs 

    def get_answer(self, sequence: str) -> str:
        """
        get the answer from the LLM generated sequence.
        """
        extracted_answer = extract_answer(sequence, mode="qa")
        if extracted_answer.strip():
            return extracted_answer
        
        if "</think>" in sequence:
            # try use the content after </think> as the answer
            extracted_answer = sequence.split("</think>")[1].strip()
            if extracted_answer.strip():
                return extracted_answer
        
        return sequence.strip()
        
    @pta.transform.by_query(add_ranks=False)
    def transform_iter(self, inp: Iterable[dict]) -> Iterable[dict]:
        sequences = [
            {
                "qid" : row['qid'],
                "question": row['query'], 
                "prompt": self.get_init_prompt(row['query'], self.multihop_qa), 
                "output": "", 
                "finished": False,
                "history": [], 
                "search_count": 0, 
                "search_queries": set(), 
                "retrieval_results": {}
            }
            for row in inp
        ]
        
        turn = 0 
        while True:
            sequences_not_finished = [seq for seq in sequences if not seq["finished"]]
            if sequences_not_finished:
                turn += 1 
                outputs: List[str] = self.generate(sequences_not_finished)

                # extract search queries
                queries = [] 
                prev_reasonings = [] 
                sequences_require_retrieval = [] 
                for seq, output in zip(sequences_not_finished, outputs):
                    seq["history"].append(output)
                    seq["prompt"] += output 
                    seq["output"] += output
                    query = self.extract_search_query(output)

                    if query is not None and seq["output"].rstrip().endswith(END_SEARCH_QUERY):

                        if seq['search_count'] < self.max_retrieval_step and query not in seq['search_queries']:
                            all_reasoning_steps = seq["output"] 
                            all_reasoning_steps = all_reasoning_steps.replace('\n\n', '\n').split("\n")
                            truncated_prev_reasoning = ""
                            for i, step in enumerate(all_reasoning_steps):
                                truncated_prev_reasoning += f"Step {i + 1}: {step}\n\n"
                            
                            prev_steps = truncated_prev_reasoning.split('\n\n')
                            if len(prev_steps) <= 5:
                                truncated_prev_reasoning = '\n\n'.join(prev_steps)
                            else:
                                truncated_prev_reasoning = ''
                                for i, step in enumerate(prev_steps):
                                    if i == 0 or i >= len(prev_steps) - 4 or BEGIN_SEARCH_QUERY in step or BEGIN_SEARCH_RESULT in step:
                                        truncated_prev_reasoning += step + '\n\n'
                                    else:
                                        if truncated_prev_reasoning[-len('\n\n...\n\n'):] != '\n\n...\n\n':
                                            truncated_prev_reasoning += '...\n\n'
                            truncated_prev_reasoning = truncated_prev_reasoning.strip('\n')

                            queries.append(query)
                            prev_reasonings.append(truncated_prev_reasoning)
                            sequences_require_retrieval.append(seq) 
                            seq["search_count"] += 1 
                            seq["search_queries"].add(query)

                        elif seq['search_count'] >= self.max_retrieval_step:
                            limit_message = f"\n{BEGIN_SEARCH_RESULT}\nThe maximum search limit is exceeded. You are not allowed to search.\n{END_SEARCH_RESULT}\n"
                            seq['prompt'] += limit_message
                            seq['output'] += limit_message
                            seq['history'].append(limit_message)

                        elif query in seq['search_queries']: 
                            limit_message = f"\n{BEGIN_SEARCH_RESULT}\nYou have searched this query. Please refer to previous results.\n{END_SEARCH_RESULT}\n"
                            seq['prompt'] += limit_message
                            seq['output'] += limit_message
                            seq['history'].append(limit_message)
                    
                    else:
                        seq["finished"] = True 

                if sequences_require_retrieval:
                    retrieval_results: List[List[dict]] = self.retrieve_docs(queries)
                    doc_analyses_outputs = self.analyze_docs(
                        sequences=sequences_require_retrieval, 
                        prev_reasonings=prev_reasonings, 
                        queries=queries, 
                        retrieval_results=retrieval_results
                    )
                    
                    doc_analyses = [item["extracted_info"] for item in doc_analyses_outputs]
                    for seq, query, retrieval_result, analysis in zip(sequences_require_retrieval, queries, retrieval_results, doc_analyses):
                        if isinstance(analysis, str):
                            text = f"\n\n{BEGIN_SEARCH_RESULT}{analysis}{END_SEARCH_RESULT}\n\n" 
                        else:
                            text = replace_recent_steps(seq['output'], analysis)
                        seq["prompt"] += text 
                        seq["output"] += text 
                        seq["history"].append(text)
                        seq["retrieval_results"][query] = retrieval_result

            unfinished = [seq for seq in sequences if not seq["finished"]]
            if not unfinished:
                break 
            else:
                if turn >= self.max_turn:
                    print(f"The maximum number of turns {self.max_turn} is exceeded, stopping...")
                    break
            
        # extract answer
        for seq in sequences:
            # seq["qanswer"] = extract_answer(seq["output"], mode="qa")
            seq["qanswer"] = self.get_answer(seq["output"])
        return sequences


class SearchO1ForceRetrieval(SearchO1):
    
    def generate(self, sequences: List[dict]) -> List[str]:

        raw_prompts = [s['prompt'] for s in sequences] 
        prompts = [s['prompt'] for s in sequences]

        reasoning_steps = [[] for _ in range(len(sequences))]
        for _ in range(3):
            input_ids, attention_mask = self.tokenizer_encode(prompts)
            token_ids = self.generator._model.generate(
                input_ids, 
                attention_mask=attention_mask,
                do_sample=True, 
                temperature=self.temperature, 
                top_p=self.top_p, 
                top_k=self.top_k,
                max_new_tokens=self.generator.max_new_tokens,
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                stopping_criteria=[StopWordCriteria(self.tokenizer, input_ids.shape[-1], ["\n\n", END_SEARCH_QUERY, self.tokenizer.eos_token])]
            )
            generated_token_ids = token_ids[:, input_ids.shape[-1]:]
            texts = self.tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)
            for i, text in enumerate(texts):
                reasoning_steps[i].append(text)
                prompts[i] += text 
        
        # filter out </think>\n\n
        new_reasoning_steps = [] 
        for steps in reasoning_steps:
            new_steps = [] 
            for step in steps:
                if step.strip().lower() == "</think>":
                    continue 
                new_steps.append(step)
            new_reasoning_steps.append(new_steps)
        reasoning_steps = new_reasoning_steps
        
        generated_texts = [''.join(steps) for steps in reasoning_steps]

        new_prompts = []
        first_step_generated_texts = []
        for i, raw_prompt in enumerate(raw_prompts):
            flag = False 
            for j in range(len(reasoning_steps[i])-1):
                if "".join(reasoning_steps[i][:j+1]).rstrip().endswith(END_SEARCH_QUERY):
                    new_prompts.append(raw_prompt + "".join(reasoning_steps[i][:j+1]))
                    first_step_generated_texts.append("".join(reasoning_steps[i][:j+1]))
                    flag = True 
                    break 
            if flag:
                continue  
            if generated_texts[i].rstrip().endswith(END_SEARCH_QUERY) or "\\boxed" in generated_texts[i] or "</think>" in generated_texts[i]:
                new_prompts.append(raw_prompt + generated_texts[i])
                first_step_generated_texts.append(generated_texts[i])
                continue            
            if "<think>" in reasoning_steps[i][0]:
                new_prompts.append(raw_prompt + "".join(reasoning_steps[i][:2]) + BEGIN_SEARCH_QUERY)
                first_step_generated_texts.append("".join(reasoning_steps[i][:2]) + BEGIN_SEARCH_QUERY)
            else:
                new_prompts.append(raw_prompt + reasoning_steps[i][0] + BEGIN_SEARCH_QUERY)
                first_step_generated_texts.append(reasoning_steps[i][0] + BEGIN_SEARCH_QUERY)
        
        additional_retrieval_required_indices = [] 
        additional_retrieval_required_prompts = [] 
        for i, prompt in enumerate(new_prompts):
            if not prompt.rstrip().endswith(END_SEARCH_QUERY):
                additional_retrieval_required_indices.append(i)
                additional_retrieval_required_prompts.append(prompt)
        generated_texts = [""] * len(new_prompts)
        if additional_retrieval_required_prompts: 
            input_ids, attention_mask = self.tokenizer_encode(additional_retrieval_required_prompts)
            token_ids = self.generator._model.generate(
                input_ids, 
                attention_mask=attention_mask,
                do_sample=True, 
                temperature=self.temperature, 
                top_p=self.top_p, 
                top_k=self.top_k,
                max_new_tokens=self.generator.max_new_tokens,
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                stopping_criteria=[StopWordCriteria(self.tokenizer, input_ids.shape[-1], [END_SEARCH_QUERY, self.tokenizer.eos_token])]
            )
            generated_token_ids = token_ids[:, input_ids.shape[-1]:]
            additional_retrieval_required_generated_texts = self.tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)
            for i, text in enumerate(additional_retrieval_required_generated_texts):
                generated_texts[additional_retrieval_required_indices[i]] = text 

        final_generated_texts = []
        for text, first_step_text in zip(generated_texts, first_step_generated_texts):
            final_generated_texts.append(first_step_text + text)
        return final_generated_texts

