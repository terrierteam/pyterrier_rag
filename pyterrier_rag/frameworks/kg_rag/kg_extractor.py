import re 
import numpy as np
from typing import Union, Optional, Tuple, List, Dict

import pandas as pd
import pyterrier as pt
from ...backend import Backend, HuggingFaceBackend 
from .prompts import generate_knowledge_triples_template, generate_knowledge_triples_chat_template


class KnowledgeGraphExtractor(pt.Transformer): #type: ignore
    """
    knowledge graph extractor class, used to extract knowledge triples from documents
    can be used in PyTerrier, supporting single document or batch document processing.
    Initially proposed in TRACE. 

    TRACE the Evidence: Constructing Knowledge-Grounded Reasoning Chains for Retrieval-Augmented Generation. Jinyuan Feng, Zaiqiao Meng and Craig Macdonald. In Proceedings of EMNLP 2024. https://arxiv.org/abs/2406.11460.
    
    Contributors: Jinyuan Feng, Jie Zhan, Craig Macdonald
    """
    
    def __init__(self, 
        backend : Backend,
        dataset="hotpotqa",
        batch_size=10,
        num_exemplars=3,
        verbose=False,
    ):
        """
        initialize the knowledge graph extractor
        
        Args:
            model_id: HuggingFace model identifier
            dataset: dataset name, used to select appropriate examples
            batch_size: batch size
            max_input_length: maximum input length
            max_new_tokens: maximum number of tokens to generate
            num_exemplars: number of examples
            device: device type ('cuda' or 'cpu')
            verbose: enable verbose logging
            backend: PyTerrier HuggingFaceBackend instance, if None will create one
        """
        super().__init__()
        self.backend = backend
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_exemplars = num_exemplars
        self.verbose = verbose

    @staticmethod
    def from_pretrained(model_id, device='cuda', max_input_length=2048, max_new_tokens=512, **kwargs) -> 'KnowledgeGraphExtractor':
        backend = HuggingFaceBackend(
                    model_id=model_id,      
                    device=device,
                    max_input_length=max_input_length,
                    max_new_tokens=max_new_tokens
        )
        return KnowledgeGraphExtractor(backend, **kwargs)

    def get_dataset_demonstrations(self, dataset):
        """get dataset demonstrations"""
        if dataset == "hotpotqa":
            from .prompts import generate_knowledge_triples_hotpotqa_exemplars
            demonstrations = generate_knowledge_triples_hotpotqa_exemplars
        elif dataset == "2wikimultihopqa":
            from .prompts import generate_knowledge_triples_2wikimultihopqa_exemplars
            demonstrations = generate_knowledge_triples_2wikimultihopqa_exemplars
        elif dataset == "musique":
            from .prompts import generate_knowledge_triples_musique_exemplars
            demonstrations = generate_knowledge_triples_musique_exemplars
        else:
            raise ValueError(f"{dataset} is not a supported dataset!")
        
        return demonstrations

    def convert_several_exemplars_to_text(self, exemplars: List[str]) -> str:
        """convert several exemplars to text format (maintaining original function)"""
        return "\n\n".join(exemplars)

    def create_prompt_for_document(self, title: str, text: str, exemplars: List[str]) -> str:
        """create a single prompt for document using exemplars"""
        
        exemplars_text = self.convert_several_exemplars_to_text(exemplars)
        
        # build prompt for HuggingFaceBackend
        if hasattr(self.backend, 'tokenizer') and hasattr(self.backend.tokenizer, 'apply_chat_template'):
            # method 1: if backend supports chat template, use chat format
            system_content = generate_knowledge_triples_chat_template.format(exemplars=exemplars_text)
            user_content = f"Title: {title}\nText: {text}\nKnowledge Triples: "
            
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            try:
                prompt = self.backend.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                if self.verbose:
                    print("Using chat template format")
                return prompt
            except Exception as e:
                if self.verbose:
                    print(f"Chat template failed: {e}, falling back to complete template")
        
        # method 2: fall back to complete template format
        prompt = generate_knowledge_triples_template.format(
            exemplars=exemplars_text,
            title=title,
            text=text
        )
        
        if self.verbose:
            print("Using complete template format")
            
        return prompt

    def get_exemplars_for_document(self, title: str, text: str) -> List[str]:
        """get exemplars for a specific document, considering context window"""
        
        dataset_demonstrations = self.get_dataset_demonstrations(self.dataset)
        # randomly select examples
        selected_indices = np.random.permutation(len(dataset_demonstrations))[:self.num_exemplars]
        
        exemplars = [dataset_demonstrations[int(idx)] for idx in selected_indices]
        exemplars = ["Title: {}\nText: {}\nKnowledge Triples: {}".format(
            example["title"], example["text"], example["triples"]) for example in exemplars]
        
        # adjust example number based on context window 
        if hasattr(self.backend, 'tokenizer'):
            final_exemplars = None
            while len(exemplars) > 0:
                for num in range(len(exemplars), 0, -1):
                    # use complete template for length test
                    exemplars_text = self.convert_several_exemplars_to_text(exemplars[:num])
                    test_prompt = generate_knowledge_triples_template.format(
                        exemplars=exemplars_text,
                        title=title,
                        text=text
                    )
                    
                    try:
                        test_tokens = self.backend.tokenizer.encode(test_prompt)
                        if len(test_tokens) <= self.max_input_length:
                            final_exemplars = exemplars[:num]
                            break
                    except Exception:
                        # if encoding fails, use estimated length
                        if len(test_prompt) <= self.max_input_length * 4:  # rough estimate
                            final_exemplars = exemplars[:num]
                            break
                if final_exemplars is None:
                    exemplars = exemplars[1:]
                else:
                    break
            
            if final_exemplars is None:
                final_exemplars = []
                if self.verbose:
                    print("Warning: No exemplars fit in context window")
            elif self.verbose:
                print(f"Selected {len(final_exemplars)} exemplars after context window optimization")
                
            return final_exemplars
        else:
            if self.verbose:
                print(f"Tokenizer not available, using all {len(exemplars)} exemplars without length check")
            return exemplars

    def parse_model_output(self, triples_text: str) -> List[Tuple]:
        """parse model output triples text"""
        results = [] 
        for one_triple_text in re.findall(r'<([^>]*)>', triples_text):
            pieces = one_triple_text.rsplit(";", maxsplit=2)
            if len(pieces) != 3:
                if self.verbose:
                    print(f"Something wrong with this triple: \"{one_triple_text}\", Skip this triple!")
                continue
            head, relation, tail = pieces
            results.append((head.strip(), relation.strip(), tail.strip()))
        return results

    def generate_triples_for_document_list(self, 
            document_list: List[Dict[str, Optional[Union[str, List[str], List[int]]]]], 
            ) -> List[List[Dict[str, Union[str, list]]]]:
        """generate triples for document list"""
        
        results = []
        import more_itertools
        for batch in more_itertools.chunked(document_list, self.batch_size):
            prompts = []
            for document in batch:
                title = document["title"]
                sentences = document["sentences"]
                text = (
                    " ".join(str(s) for s in sentences)
                    if isinstance(sentences, list)
                    else (sentences if isinstance(sentences, str) else "")
                )
                # get suitable examples
                exemplars = self.get_exemplars_for_document(title, text)
                
                # create prompt
                prompt = self.create_prompt_for_document(title, text, exemplars)
                prompts.append(prompt)

            # call the LLM
            batch_responses = self.backend.generate(prompts, max_new_tokens=512)

            for document, generated_content in zip(document_list, batch_responses):
                triples = self.parse_model_output(generated_content.text) # [(head, relation, tail)]
                document_text = " ".join(document["sentences"]) #type: ignore
                triples_in_one_document = []
                for head, relation, tail in triples:
                    if head.lower() != document["title"].lower():   #type: ignore
                        if head.lower() not in document_text.lower():
                            head = document["title"]
                    
                    triples_in_one_document.append({
                        "head": head,
                        "relation": relation, 
                        "tail": tail, 
                    })
                results.append(triples_in_one_document)

        return results

    def add_sentence_index_to_generated_triples(self,
        document_list: List[Dict[str, Optional[Union[str, List[str], List[int]]]]], 
        triples_list: List[List[Dict[str, str]]]
    )->List[List[dict]]:
        
        # document_list: [{"title": str, "sentences": [str], "ranked_prompt_indices": [int] / None}]
        # triples_list: [ [{"head": str, "relation": str, "tail": str}] ]
        def get_common_word_count(substring, text):
            return np.sum([word in text for word in substring.split()])
        
        for document, triples in zip(document_list, triples_list):
            sentences = document["sentences"]
            start_sentence_index = 0
            for triple in triples:
                triple_text = triple["relation"] + " " + triple["tail"]
                triple_sentence_common_word_count = \
                    [-100] * start_sentence_index + [get_common_word_count(triple_text, sentence) for sentence in sentences[start_sentence_index:]] #type: ignore
                index = int(np.argmax(triple_sentence_common_word_count))
                if index == start_sentence_index + 1:
                    start_sentence_index = index
                # TODO positions as sentence index might be useful in the future
                #triple["position"] = [None, index] #type: ignore
        
        return triples_list

    def transform(self, documents_df: pd.DataFrame) -> pd.DataFrame:
        """
        main interface function to extract knowledge graphs from PyTerrier pipeline
        
        Args:
            documents_df: PyTerrier input DataFrame, usually contains the following columns:
                         - 'docno': document ID
                         - 'title': document title (optional)
                         - 'text': document text content
                         - other possible columns
                       
        Returns:
            modified DataFrame, with 'knowledge_graph' column, containing knowledge triples for each document

        """
        with pt.validate.any(documents_df) as v:
            v.document_frame(extra_columns=["text"])
            v.document_frame(extra_columns=["text", "title"])
            v.result_frame(extra_columns=["text"])
            v.result_frame(extra_columns=["text", "title"])
        
        # prepare document list
        document_list = []
        for _, row in documents_df.iterrows():
            title = row.get('title', '')
            text = row['text']
            
            # split text into sentences
            if isinstance(text, str):
                sentences = text.split('. ')
                sentences = [s.strip() + '.' if not s.endswith('.') else s.strip() for s in sentences if s.strip()]
            else:
                sentences = [str(text)]
            
            document_list.append({
                "title": title,
                "sentences": sentences,
                "ranked_prompt_indices": None
            })
        
        # generate triples
        if len(documents_df):
            print(f"Extracting knowledge graphs for {len(document_list)} documents from PyTerrier pipeline...")
        document_triples_list = self.generate_triples_for_document_list(document_list)
        
        # add sentence index
        document_triples_with_sentence_index_list = self.add_sentence_index_to_generated_triples(
            document_list, document_triples_list  #type: ignore
        )
        
        # add results to DataFrame
        result_df = documents_df.copy()
        result_df['knowledge_graph'] = document_triples_with_sentence_index_list
        
        return result_df
