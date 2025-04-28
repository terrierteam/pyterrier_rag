from importlib.util import find_spec


def is_openai_availible():
    try:
        find_spec("openai")

        return True
    except ValueError:
        return False


def is_transformers_availible():
    try:
        find_spec("transformers")

        return True
    except ValueError:
        return False


def is_vllm_availible():
    try:
        find_spec("vllm")

        return True
    except ValueError:
        return False


def is_outlines_availible():
    try:
        find_spec("outlines")

        return True
    except ValueError:
        return False


def is_tiktoken_availible():
    try:
        find_spec("tiktoken")

        return True
    except ValueError:
        return False
