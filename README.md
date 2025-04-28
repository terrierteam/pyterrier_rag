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
- SearchR1 with Sparse Retrieval and MonoT5: [examples/search-r1.ipynb](https://github.com/terrierteam/pyterrier_rag/blob/stable/examples/search-r1.ipyn) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/terrierteam/pyterrier_rag/blob/stable/examples/search-r1.ipyn) 

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
monoT5_rag.search("What are chemical reactions?")
```

Try it out now with the example notebook: [sparse_retrieval_FiD_FlanT5.ipynb](https://github.com/terrierteam/pyterrier_rag/blob/stable/examples/nq/sparse_retrieval_FiD_FlanT5.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/terrierteam/pyterrier_rag/blob/stable/examples/nq/sparse_retrieval_FiD_FlanT5.ipynb).

## Agentic RAG

These frameworks use search as a tool - the reasoning model decides when to search, and then integrates the retrieved results into the input for the next invocation of the model:
 - Search-R1: `pyterrier_rag.SearchR1` https://arxiv.org/pdf/2503.09516
 - Search-O1: `pyterrier_rag.SearchO1` https://arxiv.org/abs/2501.05366
 - R1-Searcher: `pyterrier_rag.R1Searcher` https://arxiv.org/abs/2503.05592

```python
bm25 = pt.Artifact.from_hf('pyterrier/ragwiki-terrier').bm25(include_fields=['docno', 'text', 'title'])
monoT5 = pyterrier_t5.MonoT5()
r1_monoT5 = pyterrier_rag.SearchR1(bm25 % 20 >> monoT5)
r1_monoT5.search("What are chemical reactions?")

o1_monoT5 = pyterrier_rag.SearchO1(
    pyterrier_rag.readers.CausalLMReader("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"), 
    bm25 % 20 >> monoT5)
o1_monoT5.search("What are chemical reactions?")
```

Try these frameworks out now with our example notebooks: 
 - [examples/search-r1.ipynb](https://github.com/terrierteam/pyterrier_rag/blob/main/examples/search-r1.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/terrierteam/pyterrier_rag/blob/main/examples/search-r1.ipynb)
 - [examples/search-o1.ipynb](https://github.com/terrierteam/pyterrier_rag/blob/main/examples/search-r1.ipynb)
 - [examples/r1searcher.ipynb](https://github.com/terrierteam/pyterrier_rag/blob/main/examples/r1searcher.ipynb)


## Datasets

Queries and gold answers of common datasets can be accessed through the PyTerrier datasets API: `pt.get_dataset("rag:nq").get_topics()` and `pt.get_dataset("rag:nq").get_answers()`. The following QA datasets are available:

 - Natural Questions: `"rag:nq"`
 - HotpotQA: `"rag:hotpotqa"`
 - TriviaQA: `"rag:triviaqa"`
 - Musique: `"rag:musique"`
 - WebQuestions: `"rag:web_questions"`
 - WoW: `"rag:wow"`
 - PopQA: `"rag:popqa"`

We also provide pre-built indices for standard RAG corpora. For instance, a BM25 retriever for the Wikipedia corpus for NQ can be obtained from an pre-existing index autoamticallty downloaded from HuggingFace:

```python
sparse_index = pt.Artifact.from_hf('pyterrier/ragwiki-terrier')
bm25 = pt.rewrite.tokenise() >> sparse_index.bm25(include_fields=['docno', 'text', 'title']) >> pt.rewrite.reset()
```

Dense indices are also provided, e.g. E5 on Wikipedia:
```python
import pyterrier_dr
e5 = pyterrier_dr.E5() >> pt.Artifact.from_hf("pyterrier/ragwiki-e5.flex") >> sparse_index.text_loader(['docno', 'title', 'text'])
```

## Evaluation

An experiment comparing multiple RAG pipelines can be expressed using PyTerrier's pt.Experiment() API:

```python
pt.Experiment(
    [pipe1, pipe2],
    dataset.get_topics(),
    dataset.get_answers(),
    [pyterrier_rag.measures.EM, pyterrier_rag.measures.F1]
)
```

Available measures include:
 - Answer length: `pyterrier_rag.measures.AnswerLen`
 - Answers of 0 length: `pyterrier_rag.measures.AnswerZeroLen`
 - Exact match percentage: `pyterrier_rag.measures.EM`
 - F1: `pyterrier_rag.measures.F1`
 - BERTScore (measures similarity of answer with relevant documents): `pyterrier_rag.measures.BERTScore`
 - ROUGE, e.g. `pyterrier_rag.measures.ROUGE1F`

Use the `baseline` kwarg to conduct significance testing in your experiment - see the [pt.Experiment() documentation](https://pyterrier.readthedocs.io/en/latest/experiments.html) for more examples.

## Citations

If you use PyTerrier-RAG for you research, please cite our work:

_Constructing and Evaluating Declarative RAG Pipelines in PyTerrier. Craig Macdonald, Jinyuan Fang, Andrew Parry and Zaiqiao Meng. In Proceedings of SIGIR 2025._


## Credits
 - Craig Macdonald, University of Glasgow
 - Jinyuan Fang, University of Glasgow
 - Andrew Parry, University of Glasgow
 - Zaiqiao Meng, University of Glasgow
 - Sean MacAvaney, University of Glasgow
