PyTerrier-RAG How-To Guides
============================================================

This page provides task-oriented guides for common PyTerrier-RAG workflows.
The first section covers core usage; the second section covers common extensions.


Basic Workflows
----------------

.. how-to:: I want a first RAG baseline in a few lines.

    .. _pyterrier_rag:how-to:first-rag:
    .. related:: pyterrier_rag.readers.T5FiD

    Use a standard retriever followed by a FiD reader.

    .. code-block:: python
        :caption: Baseline RAG with BM25 + FiD

        import pyterrier as pt
        import pyterrier_rag as ptr

        dataset = pt.get_dataset("rag:nq")
        bm25 = pt.Artifact.from_hf("pyterrier/ragwiki-terrier").bm25(
            include_fields=["docno", "title", "text"]
        )

        fid = ptr.readers.T5FiD(
            model_name_or_path="google/flan-t5-base",
            tokenizer_name_or_path="google/flan-t5-base",
        )

        rag = bm25 % 10 >> fid
        answers = rag(dataset.get_topics("dev").head(5))


.. how-to:: I want to evaluate answer quality.

    .. _pyterrier_rag:how-to:evaluate:
    .. related:: pyterrier_rag.measures.EM
    .. related:: pyterrier_rag.measures.F1

    .. code-block:: python
        :caption: Evaluate generated answers with EM and F1

        import pyterrier as pt
        import pyterrier_rag as ptr

        dataset = pt.get_dataset("rag:nq")
        pt.Experiment(
            [rag],  # any pipeline that returns qanswer
            dataset.get_topics("dev"),
            dataset.get_answers("dev"),
            [ptr.measures.EM, ptr.measures.F1, ptr.measures.ROUGE1F],
        )


.. how-to:: I want to use a general LLM backend as the reader.

    .. _pyterrier_rag:how-to:generic-reader:
    .. related:: pyterrier_rag.Backend.from_dsn
    .. related:: pyterrier_rag.readers.Reader

    Build a backend from DSN, aggregate retrieved context, and run ``Reader``.

    .. code-block:: python
        :caption: Reader with a backend created from DSN

        import pyterrier as pt
        import pyterrier_rag as ptr

        bm25 = pt.Artifact.from_hf("pyterrier/ragwiki-terrier").bm25(
            include_fields=["docno", "title", "text"]
        )

        # Example DSN forms:
        #   huggingface:google/flan-t5-base
        #   vllm:meta-llama/Llama-3.1-8B-Instruct
        #   openai:gpt-4o-mini api_key=$OPENAI_API_KEY
        backend = ptr.Backend.from_dsn("openai:gpt-4o-mini api_key=$OPENAI_API_KEY")

        reader = ptr.readers.Reader(backend=backend)
        rag = bm25 % 5 >> ptr.prompt.Concatenator() >> reader


Extension Workflows
--------------------

.. how-to:: I want token logprobs for downstream analysis.

    .. _pyterrier_rag:how-to:logprobs:
    .. related:: pyterrier_rag.backend.TextGenerator

    .. code-block:: python
        :caption: Generate answers and token logprobs

        import pandas as pd
        import pyterrier_rag as ptr

        backend = ptr.OpenAIBackend("gpt-4o-mini", api_key="...")
        generator = backend.logprobs_generator(max_new_tokens=1)

        inp = pd.DataFrame([
            {"qid": "q1", "prompt": "What is the capital of France? Answer with one word."},
            {"qid": "q2", "prompt": "What is the capital of Italy? Answer with one word."},
        ])
        out = generator(inp)
        # out includes: qanswer and qanswer_logprobs


.. how-to:: I want to run an agentic RAG pipeline.

    .. _pyterrier_rag:how-to:agentic:
    .. related:: pyterrier_rag.SearchR1
    .. related:: pyterrier_rag.SearchO1
    .. related:: pyterrier_rag.R1Searcher

    .. code-block:: python
        :caption: Search-R1 over a standard retrieval stack

        import pyterrier as pt
        import pyterrier_rag as ptr
        import pyterrier_t5

        bm25 = pt.Artifact.from_hf("pyterrier/ragwiki-terrier").bm25(
            include_fields=["docno", "title", "text"]
        )
        mono = pyterrier_t5.MonoT5()

        search_r1 = ptr.SearchR1(bm25 % 20 >> mono)
        answers = search_r1.search("What are chemical reactions?")


.. how-to:: I want to separate final answers from reasoning traces.

    .. _pyterrier_rag:how-to:reasoning-extractor:
    .. related:: pyterrier_rag.ReasoningExtractor

    If your model emits ``<think>...</think>`` style reasoning blocks, run
    ``ReasoningExtractor`` after generation.

    .. code-block:: python
        :caption: Extract reasoning into a dedicated column

        import pandas as pd
        import pyterrier_rag as ptr

        backend = ptr.OpenAIBackend(
            "deepseek-llama-3-8b-instruct",
            api_key="...",
            base_url="http://localhost:8000/v1",
        )

        pipeline = backend >> ptr.ReasoningExtractor()
        inp = pd.DataFrame([{"qid": "q1", "prompt": "What is the capital of France?"}])
        out = pipeline(inp)
        # out includes: qanswer and reasoning
