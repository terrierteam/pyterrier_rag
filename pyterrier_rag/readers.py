from abc import ABC, abstractmethod 
from typing import Union, Tuple, Dict, List

import pyterrier as pt 
import pyterrier_alpha as pta
import torch
from typing import Iterable, Union
from transformers import AutoTokenizer, GenerationConfig
from ._fid_readers import T5FiDReader, BARTFiDReader


class Reader(pt.Transformer, ABC):

    def __init__(
        self, 
        *, 
        batch_size: int = 4,
        text_field: str = 'text',
        text_max_length: int = 512, 
        num_context: int = 5,
        max_new_tokens: int = 32, 
        generation_config: GenerationConfig = None,
        verbose: bool = False,
        device: Union[str, torch.device] = None,
        **kwargs
    ):
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device 

        self.batch_size = batch_size # TODO self.batch_size and self.verbose are not used. 
        self.text_field = text_field
        self.text_max_length = text_max_length
        self.num_context = num_context
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose
        self.kwargs = kwargs

        if generation_config is None:
            # use greedy decoding by default 
            self.generation_config = GenerationConfig(
                max_new_tokens = self.max_new_tokens, 
                temperature=1.0,
                do_sample = False, 
                num_beams = 1, 
                early_stoppint = True
            )
        else:
            self.generation_config = generation_config

    # TODO: couldn't pass self.verbose to pta.transform.by_query
    @pta.transform.by_query()
    def transform_iter(self, inp: Iterable[dict], **kwargs) -> Iterable[dict]:
        return self.transform_by_query(inp=inp, **kwargs)
    
    @abstractmethod
    def transform_by_query(self, inp: Iterable[dict], **kwargs) -> Iterable[dict]:
        pass

    def get_context_by_query(self, inp: Iterable[dict]) -> Iterable[Union[str, Tuple[str]]]:
        """
        return at most self.num_context retrieved context.
        """
        if self.num_context and inp:
            if "score" in inp[0]:
                inp = sorted(inp, key=lambda x: x["score"], reverse=True)
            if "title" in inp[0]:
                context = [(item["title"], item[self.text_field]) for item in inp]
            else:
                context = [item[self.text_field] for item in inp]
        else:
            context = None
        return context


class FiD(Reader):

    def __init__(
        self, 
        model: Union[T5FiDReader, BARTFiDReader],
        tokenizer: AutoTokenizer, 
        batch_size: int = 4, 
        text_field: str = 'text', 
        text_max_length: int = 256, 
        num_context: int = 5, 
        max_new_tokens: int = 32, 
        generation_config: GenerationConfig = None,
        verbose: bool = False, 
        device: Union[str, torch.device] = None, 
        **kwargs
    ):
        super().__init__(
            batch_size=batch_size, 
            text_field=text_field, 
            text_max_length=text_max_length, 
            num_context=num_context,
            max_new_tokens=max_new_tokens, 
            generation_config=generation_config,
            verbose=verbose, 
            device=device, 
            **kwargs
        )
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.query_prefix = "question:"
        self.title_prefix = "title:"
        self.context_prefix = "context:"

    def transform_by_query(self, inp: Iterable[dict], **kwargs) -> Iterable[dict]:

        qid = inp[0]["qid"]
        query = inp[0]["query"]
        for row in inp:
            assert row["query"] == query, "All rows must have the same query for `transform_by_query`"
        
        context = self.get_context_by_query(inp)
        input_texts = self.format_input_texts(query, context)
        inputs = self.tokenizer_encode(input_texts)
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        generated_token_ids = self.model.generate(**inputs, generation_config=self.generation_config)
        qanswer = self.tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)[0]
        
        return [ {"qid": qid, "query": query, "qanswer": qanswer} ]

    def format_input_texts(self, question: str, context: Iterable[Union[str, Tuple[str]]]) -> List[str]:

        if not context:
            return [question]
        
        input_texts = []
        for item in context:
            # append title and context prefix
            if isinstance(item, tuple):
                title, text = item 
                doc_text = self.title_prefix + " " + title + " " + self.context_prefix + " " + text
            else:
                text = item
                doc_text = self.context_prefix + " " + text 
            # prepend question 
            input_text = self.query_prefix + " " + question + " " + doc_text.strip()
            input_texts.append(input_text.strip())

        return input_texts
    
    def tokenizer_encode(self, texts: List[str]) -> Dict[str, torch.Tensor]:

        tokenizer_outputs = self.tokenizer.batch_encode_plus(
            texts, 
            max_length = self.text_max_length, 
            padding = "max_length", 
            truncation = True, 
            return_tensors = 'pt'
        )
        input_ids = tokenizer_outputs["input_ids"][None, :, :] # for only one query 
        attention_mask = tokenizer_outputs["attention_mask"][None, :, :]

        return {"input_ids": input_ids, "attention_mask": attention_mask}
    

class T5FiD(FiD):

    def __init__(self, model_name_or_path: str, tokenizer_name_or_path: str, batch_size: int = 4, text_field: str = 'text', text_max_length: int = 256, num_context: int = 5, max_new_tokens: int = 32, generation_config: GenerationConfig = None, verbose: bool = False, device: Union[str, torch.device] = None, **kwargs):
        model = T5FiDReader.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        super().__init__(model, tokenizer, batch_size, text_field, text_max_length, num_context, max_new_tokens, generation_config, verbose, device, **kwargs)


class BARTFiD(FiD):

    def __init__(self, model_name_or_path: str, tokenizer_name_or_path: str, batch_size: int = 4, text_field: str = 'text', text_max_length: int = 256, num_context: int = 5, max_new_tokens: int = 32, generation_config: GenerationConfig = None, verbose: bool = False, device: Union[str, torch.device] = None, **kwargs):
        model = BARTFiDReader.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        super().__init__(model, tokenizer, batch_size, text_field, text_max_length, num_context, max_new_tokens, generation_config, verbose, device, **kwargs)
