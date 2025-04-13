import pyterrier_rag
import pyterrier as pt
import pyterrier_rag.search_o1
reader = pyterrier_rag.readers.Seq2SeqLMReader("google/flan-t5-small")
reader.config = reader.model.config
o1 = pyterrier_rag.search_o1.SearchO1(pt.terrier.Retriever.from_dataset("vaswani", "terrier_stemmed_text"), reader)
print(o1.search("where is monterosso"))