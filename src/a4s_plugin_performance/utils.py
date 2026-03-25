from enum import Enum
from collections import defaultdict
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, Field

from a4s_plugin_interface import metric
from a4s_plugin_interface.models.measure import Measure


class _HasMetricNames(Protocol):
    @classmethod
    def metric_names(cls) -> list[str]: ...


T = TypeVar("T", bound=_HasMetricNames)


class FeatureType(str, Enum):
    INTEGER = "Integer"
    FLOAT = "Float"
    CATEGORICAL = "Categorical"
    DATE = "Date"


class Feature(BaseModel):
    name: str = Field(...)
    min: float = Field(...)
    max: float = Field(...)
    type: FeatureType = Field(...)


def merge_dicts(
    dicts: list[dict[str, dict[str, Any]]],
) -> dict[str, dict[str, list[Any]]]:
    """Merge a list of dictionaries into a single dictionary.

    Args:
        dicts (list[dict]): A list of dictionaries to merge.

    Returns:
        dict: A merged dictionary with consolidated values.

    Example:
        >>> dicts = [{"a": {"x": 1}}, {"a": {"x": 2, "y": 3}, "b": {"z": 4}}]
        >>> result = merge_dicts(dicts)
        >>> print(result)
        {'a': {'x': [1, 2], 'y': [3]}, 'b': {'z': [4]}}
    """
    merged = defaultdict(lambda: defaultdict(list))

    for d in dicts:
        for key, values in d.items():
            for k, v in values.items():
                merged[key][k].append(v)

    return {m: dict(vals) for m, vals in merged.items()}


def add_metrics(cls: type[T]) -> type[T]:
    """Class decorator that automatically adds @metric decorated export methods."""
    for name in cls.metric_names():

        @metric(name)
        def fct(
            self, evaluation_output: dict[str, dict[str, list[Any]]], _name: str = name
        ) -> list[Measure]:
            values: dict[str, list[Any]] = evaluation_output.get(_name, {})
            scores = values.get("score", [])
            dates = values.get("date", [])

            if len(scores) == 0:
                return []

            if isinstance(scores[0], (int, float)):
                return [
                    Measure(
                        name=_name,
                        score=float(score),
                        **({"time": date} if date is not None else {}),
                    )
                    for (score, date) in zip(scores, dates)
                ]
            else:
                max_i, max_j = scores[0].shape
                return [
                    Measure(
                        name=_name,
                        score=float(score[i][j]),
                        description=f"({i + 1},{j + 1})/({max_i},{max_j})",
                        **({"time": date} if date is not None else {}),
                    )
                    for (score, date) in zip(scores, dates)
                    for i in range(max_i)
                    for j in range(max_j)
                ]

        setattr(cls, f"export_metric_{name}", fct)

    return cls
