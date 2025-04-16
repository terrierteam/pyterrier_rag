# PyTerrier RAG

PyTerrier-RAG is an extension for [PyTerrier](https://github.com/terrier-org/pyterrier) that makes it easier to produce retrieval augmented generation pipelines. PyTerrier-RAG supports:
1. Easy access to common QA datasets
2. Pre-built indices for common corpora
3. Popular reader models, such as Fusion-in-Decoder, LLama
4. Evaluation measures

As well as access to all of the retrievers (sparse, learned sparse and dense) and rerankers (from MonoT5 to RankGPT) accessible through the wider [PyTerrier ecosystem](https://pyterrier.readthedocs.io/en/latest/).

Installation is as easy as `pip install git+https://github.com/terrierteam/pyterrier_rag`.

## Example Notebooks
Try it out here on Google Colab now by clicking the "Open in Colab" button!
- Sparse Retrieval with FiD and FlanT5 readers: [sparse_retrieval_FiD_FlanT5.ipynb](https://github.com/terrierteam/pyterrier_rag/blob/stable/examples/nq/sparse_retrieval_FiD_FlanT5.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/terrierteam/pyterrier_rag/blob/stable/examples/nq/sparse_retrieval_FiD_FlanT5.ipynb)

## RAG Readers

 - Fusion in Decoder: `pyterrier_rag.readers.T5FiD`, `pyterrier_rag.readers.BARTFiD`
 - OpenAI: `pyterrier_rag.readers.OpenAIReader`
 - VLLM: `pyterrier_rag.readers.VLLMReader`

RAG pipelines can be formulated as easily as:

```python
bm25 = pt.terrier.Retriever()
fid = pyterrier_rag.readers.T5FiD()
bm25_rag = bm25 % 10 >> fid 
monoT5_rag = bm25 % 10 >> MonoT5() >> fid 
```

See also the example notebooks above.

## Datasets

Queries and gold answers of common datasets can be accessed through the PyTerrier datasets API: `pt.get_dataset("rag:nq").get_topics()` and `pt.get_dataset("rag:nq").get_answers()`. The following QA datasets are available:

 - Natural Questions: `"rag:nq"`
 - HotpotQA: `"rag:hotpotqa"`
 - TriviaQA: `"rag:triviaqa"`
 - Musique: `"rag:musique"`
 - WebQuestions: `"rag:web_questions"`
 - WoW: `"rag:wow"`
 - PopQA: `"rag:popqa"`

## Evaluation

An experiment comparing multiple RAG pipelines can be expressed using PyTerrier's `pt.Experiment API`:
```python
pt.Experiment(
    [pipe1, pipe2],
    dataset.get_topics(),
    dataset.get_answers(),
    [pyterrier_rag.measures.EM, pyterrier_rag.measures.F1]
)
```

Available measures include:
 - Exact match percentage: `pyterrier_rag.measures.EM`
 - F1: `pyterrier_rag.measures.F1`
 - BERTScore (measures similarity of answer with relevant documents): `pyterrier_rag.measures.BERTScore`

## Citations

If you use PyTerrier-RAG for you research, please cite our work:

_Constructing and Evaluating Declarative RAG Pipelines in PyTerrier. Craig Macdonald, Jinyuan Fang, Andrew Parry and Zaiqiao Meng. In Proceedings of SIGIR 2025._


## Credits
 - Craig Macdonald, University of Glasgow
 - Jinyuan Fang, University of Glasgow
 - Andrew Parry, University of Glasgow
 - Zaiqiao Meng, University of Glasgow
 - Sean MacAvaney, University of Glasgow
