import pandas as pd
import pyterrier as pt
from transformers import AutoModel


def _check_imports():
    # check that spacy is installed
    try:
        from spacy.cli import download
    except ImportError:
        raise ImportError("spacy is not installed. Please install it with 'pip install spacy'")
    # ensure that the spacy model is downloaded
    download("xx_sent_ud_sm")

    try:
        import nltk
    except ImportError:
        raise ImportError("nltk is not installed. Please install it with 'pip install nltk'")
    nltk.download('punkt_tab')


class Provence(pt.Transformer):
    """
    Context pruning and reranking models from the Provence family.

    The transformer re-ranks results and prunes the document's text in a single model invocation.
    The model is designed to be used in a RAG pipeline, where the pruned text is passed to a generator model.
    However, it can also be used as a standalone reranker or pruner.

    .. cite.dblp:: conf/iclr/ChirkovaFNC25
    .. cite.dblp:: journals/corr/abs-2601-18886

    .. automethod:: provence()
    .. automethod:: xprovence_v1()
    .. automethod:: xprovence_v2()
    """

    def __init__(
        self,
        model_name: str = "naver/provence-reranker-debertav3-v1",
        *,
        batch_size: int = 32,
        disable_reranking: bool = False,
        disable_rewriting: bool = False,
        rewriting_threshold: float = 0.3,
        remove_empty: bool = True,
        device_map="auto",
    ):
        """
        Args:
            model_name: The HuggingFace name of the model to load. Should be a model from the Provence family.
            batch_size: The batch size to use when processing passages.
            disable_reranking: If True, the model will not produce reranking scores and the original order of passages will be preserved.
            disable_rewriting: If True, the model will not produce pruned passages and the original passages will be preserved.
            rewriting_threshold: The threshold to use when pruning passages. A higher threshold removes more content from passages.
            remove_empty: If True, passages that are pruned to an empty string will be removed from the output.
            device_map: The device map to use when loading the model. Passed directly to ``transformers.AutoModel.from_pretrained``.
        """
        _check_imports()
        self.model_name = model_name
        self.batch_size = batch_size
        self.disable_reranking = disable_reranking
        self.disable_rewriting = disable_rewriting
        self.rewriting_threshold = rewriting_threshold
        self.remove_empty = remove_empty
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map=device_map,
        )

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        with pt.validate.any(inp) as v:
            v.result_frame(extra_columns=["text"], mode="no_title")
            v.result_frame(extra_columns=["text", "title"], mode="with_title")

        if len(inp) == 0:
            if self.disable_reranking:
                return inp
            return inp.assign(score=[], rank=[])

        output_frames = []
        for _, group in inp.groupby("qid", sort=False):
            out = group.copy()

            results = self.model.process(
                [out.iloc[0]["query"]],
                [out["text"].tolist()],
                title="first_sentence" if v.mode == "no_title" else [out["title"].tolist()],
                batch_size=self.batch_size,
                threshold=self.rewriting_threshold,
                reorder=False,
            )

            if not self.disable_rewriting:
                out = out.assign(text=results.get("pruned_context")[0])

            if not self.disable_reranking:
                out = out.assign(score=results.get("reranking_score")[0])

            if self.remove_empty:
                out = out[out["text"] != ""]

            if not self.disable_reranking:
                pt.model.add_ranks(out, single_query=True)

            output_frames.append(out)

        return pd.concat(output_frames)

    @classmethod
    def provence(cls, **kwargs):
        """ Returns the original Provence model (``naver/provence-reranker-debertav3-v1``). kwargs are passed to the constructor.
        """
        return cls("naver/provence-reranker-debertav3-v1", **kwargs)

    @classmethod
    def xprovence_v1(cls, **kwargs):
        """ Returns the XProvence v1 model (``naver/xprovence-reranker-bgem3-v1``). kwargs are passed to the constructor.
        """
        return cls("naver/xprovence-reranker-bgem3-v1", **kwargs)

    @classmethod
    def xprovence_v2(cls, **kwargs):
        """ Returns the XProvence v2 model (``naver/xprovence-reranker-bgem3-v2``). kwargs are passed to the constructor.
        """
        return cls("naver/xprovence-reranker-bgem3-v2", **kwargs)
