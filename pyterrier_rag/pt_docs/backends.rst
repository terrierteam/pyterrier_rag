LLM Backends
=================

PyTerrier RAG supports a variety of LLM backends for generating responses. This functionality is facilitated by the
:class:`~pyterrier_rag.Backend` interface, which currently has three implementations: :class:`~pyterrier_rag.backends.HuggingFaceBackend`,
:class:`~pyterrier_rag.backends.VllmBackend`, and :class:`~pyterrier_rag.backends.OpenAIBackend`. This architecture
also allows different components to share the same backend, which is particularly useful for multi-stage RAG pipelines.


Basics
------------------------

Start by creating an instance of a backend. For example, using the :class:`~pyterrier_rag.backends.OpenAIBackend`:

.. code-block:: python
    :caption: Create an instance of a :class:`~pyterrier_rag.backends.OpenAIBackend`

    import pyterrier_rag as ptr
    backend = ptr.OpenAIBackend('gpt-4o-mini', api_key="your_openai_api_key") # or loaded from OPENAI_API_KEY environment variable, if available

The backend can be used to generate responses to prompts. For example, using the `generate` method:

.. code-block:: python
    :caption: Generate a response using the backend

    backend.generate(["What is the capital of France?"])
    # Outputs: [BackendOutput(text='The capital of France is Paris.', logprobs=None)]

Backends also function as PyTerrier Transformers. By default, they take input from the ``prompt`` column and output
to the ``qanswer`` column:

.. code-block:: python
    :caption: Generate a response using the backend

    inp = pd.DataFrame([
        {'prompt': 'What is the capital of France?'},
        {'prompt': 'What is the capital of Germany?'},
    ])
    backend(inp)
    #                             prompt                            qanswer
    # 0   What is the capital of France?    The capital of France is Paris.
    # 1  What is the capital of Germany?  The capital of Germany is Berlin.

Usually you won't use a backend directly though -- they are instead typically used by other components, such as
Prompts and Frameworks.

To set the global default backend, you can call :func:`pyterrier_rag.default_backend.set`. Note that this must be
called before any components that use the default backend are used (i.e., if you are using ``default_backend``, we
recommended setting it at the top of your script/notebook).

.. code-block:: python
    :caption: Set the default backend

    import pyterrier_rag as ptr
    ptr.default_backend.set(ptr.OpenAIBackend('gpt-4o-mini'))
    ptr.default_backend.generate(['What is the capital of France?']) # -> uses the OpenAIBackend from above

The default backend is automatically loaded via the ``PYTERRIER_RAG_DEFAULT_BACKEND`` (using :meth:`pyterrier_rag.Backend.from_dsn`)
if it is set when PyTerrier RAG is first loaded.

Readers from Backends
------------------------

.. code-block:: python
    :caption: Create a reader from a backend

    system_message = """You are an expert Q&A system that is trusted around the world. 
        Always answer the query using the provided context information,
        and not prior knowledge.
        Some rules to follow:
        1. Never directly reference the given context in your answer
        2. Avoid statements like 'Based on the context, ...' or 
        'The context information ...' or anything along those lines."""
    prompt_text = """Context information is below.
                ---------------------
                {{ qcontext }}
                ---------------------
                Given the context information and not prior knowledge, answer the query.
                Query: {{ query }}
                "Answer: """

    template = get_conversation_template("meta-llama-3.1-sp")
    prompt = PromptTransformer(
        conversation_template=template,
        system_message=system_message,
        instruction=prompt_text,
        api_type="openai"
    )

    generation_args={
        "temperature": 0.1,
        "max_tokens": 128,
    }

    # this could equally be a real OpenAI model, or a HuggingFace model, or a vLLM model, etc.
    llama = OpenAIBackend(model_name, 
                        api_key="xxx", 
                        generation_args=generation_args,
                        base_url="http://yyyy:8000/v1",)

    llama_reader = Reader(llama, prompt=prompt)
    bm25_llama = bm25_ret % 5 >> Concatenator() >> llama_reader

See :ref:`_pyterrier_rag.readers` for more information on how to use the :class:`~pyterrier_rag.readers.Reader` class with Backends.

Token Probabilities
------------------------

Some components need the log probabilities of the generated tokens (and alternative tokens). This is included
as part of the :class:`~pyterrier_rag.backend.BackendOutput` object when using `return_logprobs=True` in :meth:`~pyterrier_rag.Backend.generate`
or by using :meth:`~pyterrier_rag.Backend.logprobs_generator`. For example:

.. code-block:: python
    :caption: Include log probabilities for response

    backend.generate(["What is the capital of France?"], return_logprobs=True)
    # [BackendOutput(text='The capital of France is Paris.', logprobs=[
    #     {'The': -0.04, 'That': -0.31, ...},
    #     ...,
    #     {'Paris': -0.01, 'Berlin': -2.12, ...},
    #     ...,
    # ])]

    inp = pd.DataFrame([
        {'prompt': 'What is the capital of France?'},
        {'prompt': 'What is the capital of Germany?'},
    ])
    generator = backend.logprobs_generator()
    generator(inp)
    #                             prompt                            qanswer                           qanswer_logprobs
    # 0   What is the capital of France?    The capital of France is Paris.  [{'The': -0.04, 'That': -0.31, ...}, ...]
    # 1  What is the capital of Germany?  The capital of Germany is Berlin.  [{'The': -0.02, 'That': -0.29, ...}, ...]

This feature is typically most useful when a you have a single-token response. You can force the backend to generate
a single token using ``max_new_tokens=1`` and a suitable prompt:

.. code-block:: python
    :caption: Force a single token response

    backend.generate(["What is the capital of France? Answer in a single word only."], max_new_tokens=1, return_logprobs=True)
    # [BackendOutput(text='Paris', logprobs=[{'Paris': -0.01, 'Berlin': -2.12, ...}])]

    inp = pd.DataFrame([
        {'prompt': 'What is the capital of France? Answer in a single word only.'},
        {'prompt': 'What is the capital of Germany? Answer in a single word only.'},
    ])
    generator = backend.logprobs_generator(max_new_tokens=1)
    generator(inp)
    #                             prompt   qanswer                           qanswer_logprobs
    # 0   What is the capital of France?     Paris   [{'Paris': -0.01, 'Berlin': -2.12, ...}]
    # 1  What is the capital of Germany?    Berlin   [{'Berlin': -0.02, 'Paris': -2.29, ...}]


Reasoning
------------------------

Some models output reasoning steps (contained within a ``<think>`` tag) before the final answer. If you want to
extract these reasoning steps, you can use the ``ReasoningExtractor`` transformer in your pipeline.

.. code-block:: python
    :caption: Extract reasoning steps from a response

    from pyterrier_rag import OpenAIBackend, ReasoningExtractor

    # An example of a model that outputs reasoning steps in <think> tags:
    backend = OpenAIBackend('deepseek-llama-3-8b-instruct', api_key="your_api_key", base_url="http://localhost:8000/v1")

    pipeline = backend >> ReasoningExtractor() # extract reasoning after running the backend
    inp = pd.DataFrame([
        {'prompt': 'What is the capital of France?'},
        {'prompt': 'What is the capital of Germany?'},
    ])
    reasoning_extractor(inp)
    #                             prompt  qanswer                                                       reasoning
    # 0   What is the capital of France?    Paris    Ok, let me think about this. The capital of France is Paris.
    # 1  What is the capital of Germany?   Berlin  Ok, let me think about this. The capital of Germany is Berlin.



API Documentation
------------------------

General
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pyterrier_rag.Backend
   :members:

.. autoclass:: pyterrier_rag.backend.TextGenerator
   :members:

.. autoclass:: pyterrier_rag.backend.BackendOutput
   :members:

Implementations
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pyterrier_rag.HuggingFaceBackend
   :members:

.. autoclass:: pyterrier_rag.VLLMBackend
   :members:

.. autoclass:: pyterrier_rag.OpenAIBackend
   :members:
