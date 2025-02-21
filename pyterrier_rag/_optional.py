def is_openai_availible():
    try:
        import openai # noqa: F401
        return True
    except ImportError:
        return False

def is_vllm_availible():
    try:
        import vllm # noqa: F401
        return True
    except ImportError:
        return False

def is_outlines_availible():
    try:
        import outlines # noqa: F401
        return True
    except ImportError:
        return False

def is_tiktoken_availible():
    try:
        import tiktoken # noqa: F401
        return True
    except ImportError:
        return False
