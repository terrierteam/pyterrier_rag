import pyterrier as pt
import pyterrier_alpha as pta

def _check_imports():
    # check that spacy is installed
    try:
        import spacy
    except ImportError: 
        raise ImportError("spacy is not installed. Please install it with 'pip install spacy'")
    # ensure that the spacy model is downloaded
    from spacy.cli import download
    download("xx_sent_ud_sm")        

class XProvence(pt.Transformer):
    """
    A transformer that adds compresses document information; It uses the XProvence model to
    score the documents, and to compress the text. The compressed text replaces the text column
    and the old column is saved to the text_0 column.

    .. cite.dblp:: journals/corr/abs-2601-18886
    """ 

    def __init__(self, 
                 checkpoint="naver/xprovence-reranker-bgem3-v2", 
                 batch_size=32,
                 use_scores=True
                 ):
        _check_imports()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True, device_map="auto")
        self.batch_size = batch_size
        self.use_scores = use_scores

    @pta.transform.by_query()
    def transform(self, inp):
        with pt.validate.any(inp) as v:
            v.result_frame(extra_columns=["text"], mode='no_title')
            v.result_frame(extra_columns=["text", "title"], mode='with_title')
        # save the old text column to text_0
        inp["text_0"] = inp["text"]

        # support empty dataframe
        if len(inp) == 0:
            if self.use_scores:
                inp["score"] = []
                inp["rank"] = []
            return inp

        # invoke the model to get the compressed text and scores
        results = self.model.process(
            [inp.iloc[0]["query"]], 
            [inp["text"].values.tolist()],
            title = "first_sentence" if v.mode == "no_title" else [inp["title"].values.tolist()],
            batch_size = self.batch_size)
        # todo: what is top_k and threshold

        inp["text"] = results[0]["compressed_text"]

        if self.use_scores:
            inp["score"] = [results[0]["score"]]
            pt.model.add_ranks(inp, single_query=True)
        return inp
