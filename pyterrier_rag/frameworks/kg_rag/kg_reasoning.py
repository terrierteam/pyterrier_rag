import numpy as np
from typing import List, Dict, Literal, Optional

import pandas as pd
import pyterrier as pt
import pyterrier_dr as ptd 

import torch 
import torch.nn.functional as F
from ...backend import Backend, HuggingFaceBackend 


def _chain_to_text(chains):
    # TRACE-Triple: Use the reasoning chains directly as context
    chains_list = [] 
    for i, chain in enumerate(chains):
        for triple_item in chain["triples"]:
            triple = triple_item['triple']
            triple_sentence = triple.replace("<", "").replace(">", "").replace(";", "", 2)
            if triple_sentence not in chains_list:
                chains_list.append(triple_sentence)
    return "\n".join(chains_list)

def _chain_to_text_with_docs(chains, qid_df):
    # TRACE-Doc: Use original documents referenced by the triples in chains
    chains_documents_indices_count_dict = {}
    for i, chain in enumerate(chains):
        for triple_item in chain["triples"]:
            doc_idx, sent_idx = triple_item["triple_position"]
            if doc_idx >=0:
                chains_documents_indices_count_dict[doc_idx] = chains_documents_indices_count_dict.get(doc_idx, 0) + 1 
    
    chains_with_documents_list = []
    ranked_chains_documents_indices = sorted(chains_documents_indices_count_dict.items(), key=lambda x: x[1], reverse=True)
    for idx, count in ranked_chains_documents_indices:
        chains_with_documents_list.append("title: {}, text: {}".format(qid_df.iloc[idx].get("title", ""), qid_df.iloc[idx]["text"]))
    return "\n".join(chains_with_documents_list), len(chains_with_documents_list)

class ReasoningChainGenerator(pt.Transformer): # type: ignore
    """
    PyTerrier Transformer for generating reasoning chains from knowledge graphs
    Processes DataFrame input containing questions and knowledge graphs
    """

    def __init__(self, 
                 llm_backend : Backend,
                 ranking_model : ptd.BiEncoder,
                 model_id="meta-llama/Meta-Llama-3-8B-Instruct",
                 dataset="hotpotqa",
                 max_chain_length=4,
                 num_choices=20,
                 num_exemplars=3, # for the reasoning chain prompt
                 max_length=1024,
                 max_new_tokens=16,
                 num_beams=5,
                 num_chains=10, 
                 min_triple_prob=1e-4,
                 disable_demonstration=False,
                 calculate_ranked_prompt_indices=False,
                 trace_mode : Literal['triple', 'doc'] = 'doc',
                 device="cuda",
                 verbose=False):
        """
        Initialize the reasoning chain generator
        
        Args:
            llm_backend: PyTerrier LLM backend instance
            ranking_model: PyTerrier_DR model for computing embeddings, e.g. pyterrier_dr.E5()
            dataset: Dataset name for selecting appropriate demonstrations
            max_chain_length: Maximum length of reasoning chains
            num_choices: Number of candidate triples to consider
            num_exemplars: Number of demonstration examples
            max_length: Maximum input length for LLM
            max_new_tokens: Maximum new tokens to generate  
            num_beams: Beam search width
            num_chains: Number of chains to maintain
            min_triple_prob: Minimum probability threshold for triple selection
            disable_demonstration: Whether to disable few-shot demonstrations
            calculate_ranked_prompt_indices: Whether to rank demonstrations by similarity
            device: Computing device
            verbose: Enable verbose logging
        """
        super().__init__()
        
        # Core parameters
        self.dataset = dataset
        self.ranking_model = ranking_model
        self.max_chain_length = max_chain_length
        self.num_choices = num_choices
        self.num_exemplars = num_exemplars
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.num_chains = num_chains
        self.min_triple_prob = min_triple_prob
        self.disable_demonstration = disable_demonstration
        self.calculate_ranked_prompt_indices = calculate_ranked_prompt_indices
        self.device = torch.device(device)
        self.verbose = verbose
        self.model_id = model_id
        self.trace_mode = trace_mode
        # State variables
        self.token_id_to_choice_map = None
        
        self.llm = llm_backend
            
        if self.verbose:
            print(f"Initialized ReasoningChainGenerator with {ranking_model} ranking model")

    @staticmethod
    def from_pretrained(model_id, *args, device='cuda', max_input_length=2048, max_new_tokens=512, **kwargs) -> 'ReasoningChainGenerator':
        backend = HuggingFaceBackend(
                    model_id=model_id,      
                    device=device,
                    max_input_length=max_input_length,
                    max_new_tokens=max_new_tokens
        )
        return ReasoningChainGenerator(backend, *args, **kwargs)

    def get_dataset_demonstrations(self, dataset):
        """Get dataset-specific demonstrations"""
        if dataset == "hotpotqa":
            from .prompts import generate_reasoning_chains_hotpotqa_exemplars, reasoning_chains_hotpotqa_exemplars
            return generate_reasoning_chains_hotpotqa_exemplars, reasoning_chains_hotpotqa_exemplars
        elif dataset == "2wikimultihopqa":
            from .prompts import generate_reasoning_chains_2wikimultihopqa_exemplars, reasoning_chains_2wikimultihopqa_exemplars
            return generate_reasoning_chains_2wikimultihopqa_exemplars, reasoning_chains_2wikimultihopqa_exemplars
        elif dataset == "musique":
            from .prompts import generate_reasoning_chains_musique_exemplars, reasoning_chains_musique_exemplars
            return generate_reasoning_chains_musique_exemplars, reasoning_chains_musique_exemplars
        else:
            raise ValueError(f"{dataset} is not a supported dataset!")

    def convert_candidate_triples_to_choices(self, candidates):
        """Convert candidate triples to multiple choice format"""
        return "\n".join(["A. no need for additional knowledge triples"] + 
                        [f"{chr(ord('B')+k)}. {triple}" for k, triple in enumerate(candidates)])

    def convert_several_exemplars_to_text(self, exemplars):
        """Convert multiple examples to text format"""
        return "\n\n".join(exemplars)

    def create_reasoning_prompt(self, hop: int, question: str, existing_triples: List[str], 
                              candidate_triples: List[str], ranked_prompt_indices: Optional[List[int]] = None) -> str:
        """Create prompt for LLM to select next triple in reasoning chain (matches original logic)"""
        
        instruction = ("Select the next knowledge triple that extends an existing set of knowledge triples "
                      "to form a coherent reasoning path capable of answering a specified question. "
                      "If the current reasoning path is sufficient to answer the question, simply output A. "
                      "Please only output the choice for the next knowledge triple.")
        
        if not self.disable_demonstration:
            instruction += (f"\n\nThe followings are some examples of coherent reasoning paths capable of "
                          f"answering the specified question and how the {hop+1}-th knowledge triples in "
                          f"these paths are selected:\n\n")
            
            generate_reasoning_chains_exemplars, reasoning_chains_exemplars = self.get_dataset_demonstrations(self.dataset)
            
            # Apply ranking if provided 
            if ranked_prompt_indices is not None:
                reasoning_chains_exemplars = [reasoning_chains_exemplars[idx] for idx in ranked_prompt_indices]
                generate_reasoning_chains_exemplars = [generate_reasoning_chains_exemplars[idx] for idx in ranked_prompt_indices]

            exemplars = []
            for i, (rp_examplar, grp_examplar) in enumerate(zip(reasoning_chains_exemplars, generate_reasoning_chains_exemplars)):
                if len(grp_examplar) < hop + 1:
                    continue 
                    
                # Format exactly like original
                examplar = f"coherent reasoning path: {rp_examplar['chains']}\nquestion: {rp_examplar['question']}\n"
                examplar += f"The {hop+1}-th triple in the reasoning path is selected as:\n"
                one_step_item = grp_examplar[hop]
                examplar += (f"existing knowledge triples: {', '.join(one_step_item['triples'])}\n"
                           f"question: {one_step_item['question']}\n"
                           f"candidate knowledge triples:\n{chr(10).join(one_step_item['candidate_triples'])}\n"
                           f"the next possible triple is:{one_step_item['answer']}\n")
                exemplars.append(examplar)
                if len(exemplars) >= self.num_exemplars:
                    break
            
            # Add context window optimization logic 
            final_exemplars = exemplars  
            instruction += self.convert_several_exemplars_to_text(final_exemplars)
        else:
            instruction += "\n\n"
        
        user_content = (f"The {hop+1}-th triple in the reasoning path is selected as:\n"
                       f"existing knowledge triples: {', '.join(existing_triples)}\n"
                       f"question: {question}\n"
                       f"candidate knowledge triples:\n{self.convert_candidate_triples_to_choices(candidate_triples)}\n"
                       f"the next possible triple is:")
        
        # Try to use chat template if available 
        try:
            if self.llm is not None and hasattr(self.llm, 'tokenizer') and hasattr(self.llm.tokenizer, 'apply_chat_template'):
                messages = [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": user_content}
                ]
                prompt = self.llm.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                return prompt
        except Exception:
            pass
        
        # Fallback to simple format
        return f"System: {instruction}\n\nUser: {user_content}\n\nAssistant:"

    def generate_reasoning_chains_for_question(self, question: str, triples: List[str], 
                                             triple_positions: List, 
                                             ranked_prompt_indices: Optional[List[int]] = None) -> List[Dict]:
        """Generate reasoning chains for a single question using beam search"""
        
        if not triples:
            return []
        
        # Compute embeddings for all knowledge triples
        num_total_triples = len(triples)
        triples_embeddings = np.stack([rec['doc_vec'] for rec in self.ranking_model([{'docno' : 't' + str(i), 'text' : triple } for i, triple in enumerate(triples)])])
        # Initialize beam search with empty paths
        paths = [[]]
        paths_scores = [1.0]
        paths_finished = [False]
        
        # Iteratively construct reasoning paths
        for j in range(self.max_chain_length):
            # Continue if we have active paths (unfinished)
            if all(paths_finished):
                break
                
            # Create query representations from current paths and question
            queries = [
                f"knowledge triples: {' '.join([triples[idx] for idx in path])}\nquestion: {question}"
                for path in paths
            ]
            
            # Get embeddings for current path + question combinations
            queries_embeddings = np.stack([rec['query_vec'] for rec in self.ranking_model([{'qid' : 't' + str(i), 'query' : q } for i, q in enumerate(queries)])]) 

            # Calculate similarity between queries and all triples
            queries_triples_similarities = np.matmul(queries_embeddings, triples_embeddings.T)

            # Mask out triples already used in each path
            candidate_triples_mask = np.ones_like(queries_triples_similarities)
            for k, path in enumerate(paths):
                candidate_triples_mask[k, path] = 0.0

            # set similarities for already consumed triples to be very negative 
            almost_minus_inf = np.finfo(queries_triples_similarities.dtype).min
            queries_triples_similarities = (queries_triples_similarities + 
                                           almost_minus_inf * (1.0 - candidate_triples_mask))
              
            # Select top-k most relevant triples for each path
            topk_most_relevant_triples_indices = torch.topk(
                torch.from_numpy(queries_triples_similarities), 
                k=min(self.num_choices, num_total_triples), # TODO : perhaps we can be more conservative here to prevent triple reuse
                dim=1
            )[1].tolist()

            # Process each path to generate next choices (batched generation)
            choices_and_probs = [None] * len(paths)
            prompts = []
            prompt_to_path = []

            for i, (path, path_score, finished) in enumerate(zip(paths, paths_scores, paths_finished)):
                if finished:
                    choices_and_probs[i] = [('A', 1.0)]  # Finished path
                    continue

                # Get candidate triples for this path
                candidate_indices = topk_most_relevant_triples_indices[i]
                candidate_triples = [triples[idx] for idx in candidate_indices]
                existing_triples = [triples[idx] for idx in path]

                # Create prompt for LLM and add to batch
                prompt = self.create_reasoning_prompt(
                    hop=j,
                    question=question,
                    existing_triples=existing_triples,
                    candidate_triples=candidate_triples,
                    ranked_prompt_indices=ranked_prompt_indices
                )
                prompts.append(prompt)
                prompt_to_path.append(i)

            # If we have any prompts, run a single batched generation
            if len(prompts) > 0:
                responses = self.llm.generate(prompts, return_logprobs=True, max_new_tokens=1)
                for resp, path_idx in zip(responses, prompt_to_path):
                    logprob_map = resp.logprobs[0]
                    # Sort token logits descending and compute softmax probs
                    sorted_logits, sorted_choices = zip(*sorted(zip(logprob_map.values(), logprob_map.keys()), reverse=True))
                    sorted_probs = F.softmax(torch.Tensor(sorted_logits), dim=0)

                    # Build choice list respecting available tokens
                    path_choices = []
                    max_take = min(self.num_beams, len(sorted_choices))
                    for b in range(max_take):
                        path_choices.append((sorted_choices[b], sorted_probs[b].item()))

                    choices_and_probs[path_idx] = path_choices

            # Ensure every path has a choices entry (safety)
            for i in range(len(choices_and_probs)):
                if choices_and_probs[i] is None:
                    choices_and_probs[i] = []

            # Extend paths based on choices (beam search like original)
            new_paths, new_paths_scores, new_paths_finished = [], [], []
            
            for i in range(len(paths)):
                if paths_finished[i]:
                    new_paths.append(paths[i])
                    new_paths_scores.append(paths_scores[i])
                    new_paths_finished.append(True)
                    continue
                
                path_choices = choices_and_probs[i]
                candidate_indices = topk_most_relevant_triples_indices[i]
                
                # Process each choice for this path (beam expansion)
                for choice, choice_prob in path_choices:
                    if choice_prob < self.min_triple_prob:
                        continue
                        
                    if choice == 'A':
                        # Path is complete  
                        new_paths.append(paths[i] + [-1])
                        new_paths_scores.append(paths_scores[i] * choice_prob)
                        new_paths_finished.append(True)
                    elif len(choice) == 1: # skip any non-character generations in logprobs
                        # Add selected triple to path
                        choice_idx = ord(choice) - ord('B')
                        if 0 <= choice_idx < len(candidate_indices):
                            new_paths.append(paths[i] + [candidate_indices[choice_idx]])
                            new_paths_scores.append(paths_scores[i] * choice_prob)
                            new_paths_finished.append(False)
            
            # Select top-k paths by score for next iteration (beam search)
            if len(new_paths) > self.num_chains:
                path_indices = sorted(range(len(new_paths_scores)), key=lambda x: new_paths_scores[x], reverse=True)
                top_indices = path_indices[:self.num_chains]
                paths = [new_paths[idx] for idx in top_indices]
                paths_scores = [new_paths_scores[idx] for idx in top_indices]
                paths_finished = [new_paths_finished[idx] for idx in top_indices]
            else:
                paths = new_paths
                paths_scores = new_paths_scores
                paths_finished = new_paths_finished
        
        # Build final reasoning chains (matching original format)
        reasoning_chains = []  
        for path, path_score in zip(paths, paths_scores):
            chain_triples = []
            for triple_index in path:
                if triple_index >= 0:  # -1 indicates end of chain
                    chain_triples.append({
                        "triple": triples[triple_index],
                        "triple_position": triple_positions[triple_index] if triple_index < len(triple_positions) else None
                    })
            
            reasoning_chains.append({
                "triples": chain_triples,
                "score": path_score
            })
        
        # Ensure we always return at least one chain (even if empty)
        if not reasoning_chains:
            reasoning_chains.append({
                "triples": [],
                "score": 0.1
            })
        
        return reasoning_chains

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame by adding reasoning chains to each row
        
        Expected input DataFrame columns:
        - 'query': The question to answer
        - 'knowledge_graph': Knowledge graph triples (from kg_extractor)
        - Optional: 'ranked_prompt_indices': Pre-computed demonstration rankings
        
        Output DataFrame: Input + 'reasoning_chains' column
        """
        with pt.validate.any(df) as v:
            v.result_frame(extra_columns=["text", "query", "knowledge_graph"])
            v.result_frame(extra_columns=["text", "title", "query", "knowledge_graph"])
    
        if len(df) == 0:
            cols = pt.model.query_columns(df)+["qcontext"]
            if self.trace_mode == 'doc':
                cols += ['qcontext_doc_count']
            return pd.DataFrame([], columns=cols)

        result_rows = []
        
        # Add progress bar for processing questions
        # iter = df.iterrows()
        # if self.verbose:
        #     iter = pt.tqdm(iter, desc="Processing Questions")

        for idx, qid_group in df.groupby("qid"):
            # Extract question
            question = qid_group.iloc[0]['query']

            # Extract knowledge graph triples
            triples, triple_positions = [], []

            for doc_rank, doc_kg in enumerate(qid_group["knowledge_graph"]):
                for kg_item in doc_kg:
                    if isinstance(kg_item, dict):
                        triple_text = f"<{kg_item.get('head', '')}; {kg_item.get('relation', '')}; {kg_item.get('tail', '')}>"
                        triples.append(triple_text)
                        triple_positions.append((doc_rank, kg_item.get('position', None)))
                    elif isinstance(kg_item, list):
                        # Handle nested list format from kg_extractor
                        for triple_item in kg_item:
                            if isinstance(triple_item, dict):
                                triple_text = f"<{triple_item.get('head', '')}; {triple_item.get('relation', '')}; {triple_item.get('tail', '')}>"
                                triples.append(triple_text)
                                triple_positions.append((doc_rank, triple_item.get('position', None)))

            
            # Get ranked prompt indices if available
            ranked_prompt_indices = None # TODO: row.get('ranked_prompt_indices', None)
            
            # Generate reasoning chains
            reasoning_chains = self.generate_reasoning_chains_for_question(
                question=question,
                triples=triples,
                triple_positions=triple_positions,
                ranked_prompt_indices=ranked_prompt_indices
            )
            
            # Add results to output row
            output_for_qid = qid_group.head(1)[pt.model.query_columns(qid_group)]
            if self.trace_mode == 'triple':
                output_for_qid['qcontext'] = _chain_to_text(reasoning_chains) 
            elif self.trace_mode == 'doc':
                qcontext, num_docs = _chain_to_text_with_docs(reasoning_chains, qid_group)
                output_for_qid['qcontext'] = qcontext
                output_for_qid['qcontext_doc_count'] = num_docs
                 
            result_rows.append(output_for_qid)

        # we have all the results, now return the final df
        return pd.concat(result_rows)