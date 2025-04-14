import pyterrier_rag
import pyterrier as pt
import pyterrier_rag.search_o1

import torch 

# reader = pyterrier_rag.readers.Seq2SeqLMReader("google/flan-t5-small")
# reader.config = reader.model.config
# o1 = pyterrier_rag.search_o1.SearchO1(pt.terrier.Retriever.from_dataset("vaswani", "terrier_stemmed_text"), reader)
# print(o1.search("where is monterosso"))

model_args = {"torch_dtype": torch.bfloat16, "trust_remote_code": True}
generator = pyterrier_rag.readers.CausalLMReader(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", 
    model_args = model_args, 
    text_max_length=4096, 
    max_new_tokens=2048, 
    
)

# model = pyterrier_rag.search_o1.SearchO1ForceRetrieval(
model = pyterrier_rag.search_o1.SearchO1(
    retriever = pt.terrier.Retriever.from_dataset("vaswani", "terrier_stemmed_text"),
    generator = generator,
    multihop_qa=True 
)

outputs = model.transform_iter([{"qid": "0", "query": "Which film was released earlier, Spy Of Napoleon or The Hairdresser'sq Husband?"}])
print("Reasoning Process: ", outputs[0]["output"])
print("Predicted Answer: ", outputs[0]["qanswer"]) 

# print(model.search("where is monterosso"))