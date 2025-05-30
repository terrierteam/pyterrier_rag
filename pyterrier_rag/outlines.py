from pyterrier_rag._optional import is_outlines_availible
from functools import partial
from typing import List, Any, Union, Callable, Optional


availible = is_outlines_availible()

try:
    import interegular
except:
    pass

if availible:
    from outlines.types import Regex


def choice(choices: List[Any]):
    assert availible, "Outlines not found in environment"
    from outlines.generate import choice
    return partial(choice, choices=choices)


def regex(regex_str: str | Regex):
    assert availible, "Outlines not found in environment"
    from outlines.generate import regex
    return partial(regex, regex_str=regex_str)


def json(schema_object: Union[str, object, Callable], whitespace_pattern: Optional[str] = None):
    assert availible, "Outlines not found in environment"
    from outlines.generate import json
    return partial(json,
                   schema_object=schema_object,
                   whitespace_pattern=whitespace_pattern)


def cfg(cfg_str: str):
    assert availible, "Outlines not found in environment"
    from outlines.generate import cfg
    return partial(cfg, cfg_str=cfg_str)


def format(python_type):
    assert availible, "Outlines not found in environment"
    from outlines.generate import format
    return partial(format, python_type=python_type)


def fsm(fsm: interegular.fsm.FSM):
    assert availible, "Outlines not found in environment"
    from outlines.generate import fsm
    return partial(fsm, fsm=fsm)
