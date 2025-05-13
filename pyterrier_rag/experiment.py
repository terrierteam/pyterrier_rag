from warnings import warn
import os
import sys
import pandas as pd
import numpy as np
from typing import Callable, Iterator, Union, Dict, List, Tuple, Sequence, Any, Literal, Optional, overload, IO
import types
from pyterrier import Transformer
from pyterrier.model import coerce_dataframe_types
from pyterrier._ops import Compose
import ir_measures
import tqdm as tqdm_module
from ir_measures import Measure, Metric
import pyterrier as pt

# Convenience type aliases retained from original implementation
SYSTEM_OR_RESULTS_TYPE = Any
MEASURES_TYPE = Sequence[Any]
TEST_FN_TYPE = Any
SAVEMODE_TYPE = str
SAVEFORMAT_TYPE = Union[str, Tuple[Any, Any]]

MEASURE_TYPE=Union[str, Measure]
MEASURES_TYPE=Sequence[MEASURE_TYPE]
SAVEMODE_TYPE=Literal['reuse', 'overwrite', 'error', 'warn']

SYSTEM_OR_RESULTS_TYPE = Union[Transformer, pd.DataFrame]
SAVEFORMAT_TYPE = Union[Literal['trec'], types.ModuleType, Tuple[Callable[[IO], pd.DataFrame], Callable[[pd.DataFrame, IO], None]]]

NUMERIC_TYPE = Union[float,int,complex]
TEST_FN_TYPE = Callable[ [Sequence[NUMERIC_TYPE],Sequence[NUMERIC_TYPE]], Tuple[Any,NUMERIC_TYPE] ]


def _split_metrics_by_source(metrics: MEASURES_TYPE) -> Tuple[List[Any], List[Any]]:
    """Return two disjoint lists: (metrics_needing_qrels, metrics_needing_nuggets).

    Any metric whose ``__module__`` starts with ``"pyterrier_rag.nuggetizer"`` is
    assumed to consume *nuggets* instead of *qrels*.
    """
    qrels_metrics, nugget_metrics = [], []
    for m in metrics:
        # String metrics (e.g. "map") are always qrels-driven
        if isinstance(m, str):
            qrels_metrics.append(m)
            continue
        mod = getattr(m, "__module__", "")
        if mod.startswith("pyterrier_rag.nuggetizer"):
            nugget_metrics.append(m)
        else:
            qrels_metrics.append(m)
    return qrels_metrics, nugget_metrics


def _merge_measure_dicts(preferred: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    """Helper that combines two measure dictionaries giving *preferred* precedence.

    When two dictionaries contain the same metric name the values found in
    *preferred* are kept. This makes the merge idempotent when the same metric
    appears in both lists (defensive programming – should not normally happen).
    """
    merged = preferred.copy()
    for k, v in other.items():
        if k not in merged:
            merged[k] = v
    return merged


# ---------------------------------------------------------------------------
#  Modified Experiment
# ---------------------------------------------------------------------------

def Experiment(
    retr_systems: Sequence[SYSTEM_OR_RESULTS_TYPE],
    topics: pd.DataFrame,
    qrels: pd.DataFrame,
    nuggets: pd.DataFrame,
    eval_metrics: MEASURES_TYPE,
    names: Optional[Sequence[str]] = None,
    perquery: bool = False,
    dataframe: bool = True,
    batch_size: Optional[int] = None,
    filter_by_qrels: bool = False,
    filter_by_topics: bool = True,
    baseline: Optional[int] = None,
    test: Union[str, TEST_FN_TYPE] = "t",
    correction: Optional[str] = None,
    correction_alpha: float = 0.05,
    highlight: Optional[str] = None,
    round: Optional[Union[int, Dict[str, int]]] = None,
    verbose: bool = False,
    save_dir: Optional[str] = None,
    save_mode: SAVEMODE_TYPE = "warn",
    save_format: SAVEFORMAT_TYPE = "trec",
    precompute_prefix: bool = False,
    **kwargs,
):
    """Drop-in replacement for :pyfunc:`pt.Experiment` with *nuggetised* metrics.

    The signature is identical to the canonical implementation except for the
    additional *nuggets* argument and the automatic routing of evaluation
    measures originating from ``pyterrier_rag.nuggetizer`` to that dataframe.
    All non-nugget metrics are evaluated against *qrels* as usual.
    """

    if not isinstance(retr_systems, list):
        raise TypeError(
            "Expected list of transformers for retr_systems, instead received %s" % str(type(retr_systems))
        )
    if len(kwargs):
        raise TypeError("Unknown kwargs: %s" % (str(list(kwargs.keys()))))
    if baseline is not None:
        assert int(baseline) >= 0 and int(baseline) < len(retr_systems)
        assert not perquery

    # Load topics/qrels/nuggets from disk paths if required
    if isinstance(topics, str) and os.path.isfile(topics):
        topics = pt.io.read_topics(topics)
    if isinstance(qrels, str) and os.path.isfile(qrels):
        qrels = pt.io.read_qrels(qrels)
    if isinstance(nuggets, str) and os.path.isfile(nuggets):
        nuggets = pt.io.read_qrels(nuggets)  # nuggets share the qrels schema

    if filter_by_qrels:
        topics = topics.merge(qrels[["qid"]].drop_duplicates())
        if len(topics) == 0:
            raise ValueError(
                "There is no overlap between the qids found in the topics and qrels. If this is intentional, set filter_by_topics=False and filter_by_qrels=False."
            )
    if filter_by_topics:
        qrels = qrels.merge(topics[["qid"]].drop_duplicates())
        nuggets = nuggets.merge(topics[["qid"]].drop_duplicates())
        if len(qrels) == 0:
            raise ValueError(
                "There is no overlap between the qids found in the topics and qrels. If this is intentional, set filter_by_topics=False and filter_by_qrels=False."
            )

    # ---------------------------------------------------------------------
    #  NEW: Split metrics by their data source
    # ---------------------------------------------------------------------
    qrels_metrics, nugget_metrics = _split_metrics_by_source(eval_metrics)

    # ---------------------------------------------------------------------
    #  Remainder of implementation: identical control flow except that each
    #  system is evaluated up to twice (once per judgment set) and the results
    #  are merged before significance testing/high-level post-processing.
    # ---------------------------------------------------------------------

    all_topic_qids = topics["qid"].values
    evalsRows = []
    evalDict = {}
    evalDictsPerQ = []
    actual_metric_names: List[str] = []
    mrt_needed = False
    if "mrt" in eval_metrics:
        mrt_needed = True
    #  Pre-compute common prefixes as before
    precompute_time, execution_topics, execution_retr_systems = pt.experiment._precomputation(  # type: ignore
        retr_systems, topics, precompute_prefix, verbose, batch_size
    )

    from scipy import stats

    if isinstance(test, str):
        test_fn = stats.ttest_rel if test == "t" else stats.wilcoxon
    else:
        test_fn = test  # Callable supplied by caller

    if names is None:
        names = [str(system) for system in retr_systems]
    elif len(names) != len(retr_systems):
        raise ValueError("names should be the same length as retr_systems")

    # tqdm construction unchanged
    tqdm_args = dict(disable=not verbose, unit="system", total=len(retr_systems), desc="pt.Experiment")
    if batch_size is not None:
        import math

        tqdm_args.update(unit="batches", total=math.ceil(len(topics) / batch_size) * len(retr_systems))

    with pt.tqdm(**tqdm_args) as pbar:  # type: ignore
        for name, system in zip(names, execution_retr_systems):
            # Determine save_file path once – reused across both evaluations so that
            # the run happens at most once on disk.
            save_file = None
            if save_dir is not None:
                if save_format == "trec":
                    save_ext = "res.gz"
                elif isinstance(save_format, types.ModuleType):
                    save_ext = "mod"
                elif isinstance(save_format, tuple):
                    save_ext = "custom"
                else:
                    raise ValueError("Unrecognised save_mode %s" % str(save_format))
                save_file = os.path.join(save_dir, f"{name}.{save_ext}")

            # ----------------------------------------------------------
            #  1. Metrics evaluated against qrels
            # ----------------------------------------------------------
            combined_measures: Dict[str, Any] = {}
            run_time: Optional[float] = None
            for metrics_subset, judgments in (
                (qrels_metrics, qrels),
                (nugget_metrics, nuggets),
            ):
                if not metrics_subset:
                    continue  # skip empty split

                # The first invocation performs the run; subsequent invocations reuse the save file
                mode_for_this_call = save_mode if run_time is None else "reuse"
                time_taken, measured = pt.experiment._run_and_evaluate(  # type: ignore
                    system,
                    execution_topics,
                    judgments,
                    metrics_subset,
                    perquery=perquery or baseline is not None,
                    batch_size=batch_size,
                    backfill_qids=all_topic_qids if perquery else None,
                    save_file=save_file,
                    save_mode=mode_for_this_call,
                    save_format=save_format,
                    pbar=pbar,
                )
                run_time = run_time or time_taken  # preserve first measurement
                combined_measures = _merge_measure_dicts(combined_measures, measured)

            # From here on the original logic (mrt, per-query handling, etc.)
            if baseline is not None:
                evalDictsPerQ.append(combined_measures)
                combined_measures = pt.experiment._mean_of_measures(combined_measures)  # type: ignore

            if perquery:
                for qid, mvals in combined_measures.items():
                    for measurename, val in mvals.items():
                        evalsRows.append([name, qid, measurename, val])
                evalDict[name] = combined_measures
            else:
                if mrt_needed:
                    run_time = (run_time or 0.0) + precompute_time
                    combined_measures["mrt"] = run_time / float(len(all_topic_qids))
                actual_metric_names = list(combined_measures.keys())
                evalsRows.append([name] + [combined_measures[m] for m in actual_metric_names])
                evalDict[name] = [combined_measures[m] for m in actual_metric_names]

    # The remainder of the function (dataframe construction, significance testing,
    # styling, etc.) is unchanged from the original implementation and can be
    # reused verbatim by calling the internal helper. To avoid duplicating ~200
    # lines of boilerplate we delegate to the untouched downstream code path.
    return pt.experiment._post_process_experiment_output(  # type: ignore
        evalsRows,
        evalDict,
        actual_metric_names,
        perquery,
        dataframe,
        names,
        baseline,
        test_fn,
        correction,
        correction_alpha,
        round,
        highlight,
        mrt_needed,
        evalDictsPerQ,
    )
