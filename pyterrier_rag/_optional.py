def is_openai_availible():
    try:
        import openai
        return True
    except ImportError:
        return False

def is_vllm_availible():
    try:
        import vllm
        return True
    except ImportError:
        return False

def is_outlines_availible():
    try:
        import outlines
        return True
    except ImportError:
        return False

def is_tiktoken_availible():
    try:
        import tiktoken
        return True
    except ImportError:
        return False