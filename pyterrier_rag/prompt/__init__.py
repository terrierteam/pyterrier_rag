from .jinja import jinja_formatter
from .default import CONCAT_DOCS, DefaultPrompt

# PromptTransformer is deprecated - use Reader with prompt templates instead
# This stub is provided for backwards compatibility with old tests
class PromptTransformer:
    """Deprecated: Use Reader with prompt templates instead."""
    def __init__(self, **kwargs):
        raise DeprecationWarning("PromptTransformer is deprecated. Use Reader with prompt templates instead. See the documentation for more details.")
        self.kwargs = kwargs

__all__ = ["jinja_formatter", "CONCAT_DOCS", "DefaultPrompt", "PromptTransformer"]
