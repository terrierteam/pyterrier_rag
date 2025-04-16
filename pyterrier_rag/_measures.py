import string
from collections import Counter
from typing import List

import regex
import ir_measures
import pyterrier_alpha as pta


# Normalization from SQuAD evaluation script https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
def normalize_answer(s : str) -> str:
    def remove_articles(text: str) -> str:
        return regex.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())
    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def exact_match_score(prediction : str, ground_truth : str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def ems(prediction : str, ground_truths : List[str]) -> float:
    return max([exact_match_score(prediction, gt) for gt in ground_truths])

# F1 Evaluation from HotPotQA evaluation script: https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_evaluate_v1.py
def f1_score(prediction : str, ground_truth : List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    ZERO_METRIC = (0, 0, 0)
    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC[0]
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC[0]
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC[0]
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    # return f1, precision, recall
    return f1

# we aggregate across multiple gold_answer values using max().
F1 = ir_measures.define_byquery(
    lambda qrels, res: max([f1_score(res.iloc[0]['qanswer'], gold) for gold in qrels['gold_answer']]), support_cutoff=False, name="F1")
# ems function handles the max()
EM = ir_measures.define_byquery(
    lambda qrels, res: ems(res.iloc[0]['qanswer'], qrels['gold_answer']), support_cutoff=False, name="EM")

_bertscore_model = None
def _bertscore(qrels, res, rel = 3, submeasure='f1', agg='max'):

    pta.validate.columns(qrels, includes=['query_id', 'relevance', 'text'])
    pta.validate.columns(res, includes=['query_id', 'qanswer'])
    
    assert len(res), "Empty res df provided"
    assert len(qrels), "Empty qrels df provided"
    qrels = qrels[qrels.relevance >= rel]
    assert len(qrels), "No qrels found with minimum label of %d" % rel

    global _bertscore_model
    if _bertscore_model is None:
        from evaluate import load # this is a huggingface package
        _bertscore_model = load("bertscore")
    
    predictions = res['qanswer'].to_list()
    assert len(predictions) == 1, "Unexpected number of predictions"
    references = qrels['text'].to_list()
    # duplicate the prediction for the nbr of ground truths 
    predictions = predictions * len(references)

    results = _bertscore_model.compute(predictions=predictions, references=references, lang="en", model_type="bert-large-uncased", verbose=False)
    precisions, recall, f1 = results['precision'], results['recall'], results['f1']
    r = {
        'precision': {'avg': sum(precisions)/len(precisions), 'max': max(precisions)},
        'recall': {'avg': sum(recall)/len(recall), 'max': max(recall)},
        'f1': {'avg': sum(f1)/len(f1), 'max': max(f1)},
        }
    return r[submeasure][agg]

def BERTScore(rel=3, submeasure : str = 'f1', agg : str = 'max'):
    '''
    Implements BERTScore, a semantic measure of equivalence. This is defined to take a qrels dataframe with an additional text attribute,
    and compare with the generated qanswers. 

    Arguments:
     - rel(int): Minimum label value for relevant qrels. Defaults to 3, which is the highest label in MSMARCO.
     - submeasure(str): One of 'precision', 'recall' and 'f1'. Defaults to 'f1'.
     - agg(str): How to combine (aggregate) when there are multiple relevant documents. Valid options are 'max' or 'avg'. Defaults to 'max'.

    Returns:
     An IR measures measure object that can be used in pt.Evaluate or pt.Experiment
    '''
    return ir_measures.define_byquery( lambda qrels, res: _bertscore(qrels, res, rel=rel, agg=agg), name='BERTScore', support_cutoff=False)