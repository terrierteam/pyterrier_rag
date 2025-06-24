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

    impirt pyterrier_rag as ptr
    backend = ptr.OpenAIBackend(api_key="your_openai_api_key") # or loaded from OPENAI_API_KEY environment variable, if available

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


API Documentation
------------------------

.. autoclass:: pyterrier_rag.Backend
   :members:

.. autoclass:: pyterrier_rag.HuggingFaceBackend
   :members:

.. autoclass:: pyterrier_rag.VllmBackend
   :members:

.. autoclass:: pyterrier_rag.OpenAIBackend
   :members:
