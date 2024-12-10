

class FiD(pt.Transformer):
  def __init__(self, model, tok):
    self.model = model
    self.tok = tok

  def transform_iter(self, inp):
    question = inp[0]["query"]
    qid = inp[0]["qid"]

    # TODO remove this when we have the pta.transform_iter.by_query() decorator
    for for row in inp:
      assert row["query"] == question 
    
    docs = [row["text"] for row in inp]
    # TODO tokenizer

    # TODO call self.model
    # put answer in answer
    return [ { 'qid' : qid, 'query' : question, 'qanswer' : answer} ]
    

class T5FiD(FiD):
  def __init__(self):
    from .fid_readers import T5FidReader
    super().__init__(T5FidReader.from_pretrained(TODO_name), AutoTokenizer.from_pretrained('t5')

