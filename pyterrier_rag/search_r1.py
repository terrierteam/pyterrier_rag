import transformers
import torch
import pyterrier as pt, pyterrier_alpha as pta
import pyterrier.model

# we'll need this to load the model using device_map, so best check its installed.
import accelerate

# Define the custom stopping criterion
class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False

R1_PROMPT = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: """

curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

def get_query(text):
    import re
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None
    
def get_answer(text):
    import re
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()
    else:
        return None


def search(search_pipeline : pt.Transformer, query: str, qid='1', top_k = None) -> str:
    if top_k is not None:
        search_pipeline = search_pipeline % top_k
    
    res = search_pipeline.search(query, qid=qid)
          
    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result.itertuples()):
            content = doc_item.text
            title = doc_item.title
            format_reference += f"Doc {idx+1}(Title: {title}) {content}\n"
        return format_reference

    return _passages2string(res)

class SearchR1(pt.Transformer):
    """
    Implements the SearchR1 model as a PyTerrier transformer.
    Uses code from https://github.com/PeterGriffinJin/Search-R1.

    Input columns:
     - qid
     - query
    
    Output columns:
     - qid
     - query
     - output (output of the model)
     - answer (extracted from the output)
    """

    def __init__(self, 
                 retriever : pt.Transformer, 
                 model_id : str = 'PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo',
                 retrieval_top_k = 3):
        """
        Instantiates the SearchR1 model using the specified retriever. 

        Arguments:
         - retriever(pt.Transformer): Should receive qid and query columns, and provide text, title and rank columns.
         - model_id(str): Which HGF model. Defaults to  'PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo'
         - retrieval_top_k3(int): How many documents. Defaults to 3. Use None for as many as the retriever will provide.
        """
        self.retriever = retriever
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        # Initialize the tokenizer and model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
        self.curr_eos = [151645, 151643] # for Qwen2.5 series models
        self.retrieval_top_k = retrieval_top_k

        # Initialize the stopping criteria
        target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
        self.stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, self.tokenizer)])

    @pta.transform.by_query(add_ranks=False)
    def transform_iter(self, inp : pyterrier.model.IterDict) -> pyterrier.model.IterDict:
        inp = next(inp)
        cnt = 0
        question = inp['query']
        qid = inp['qid']
        question = question.strip()
        if question[-1] != '?':
            question += '?'
        
        prompt = R1_PROMPT + question + "\n"
        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)

        while True:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            attention_mask = torch.ones_like(input_ids)
            
            # Generate text with the stopping criteria
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1024,
                stopping_criteria=self.stopping_criteria,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7
            )

            if outputs[0][-1].item() in self.curr_eos:
                generated_tokens = outputs[0][input_ids.shape[1]:]
                output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                answer = get_answer(output_text)
                return [{'qid' : qid, 'query' : question, 'answer' : answer, 'output': output_text}]

            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            tmp_query = get_query(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
            if tmp_query:
                search_results = search(self.retriever, tmp_query, qid="%s-%d" % (qid, cnt))
            else:
                search_results = ''

            search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
            prompt += search_text
            cnt += 1
