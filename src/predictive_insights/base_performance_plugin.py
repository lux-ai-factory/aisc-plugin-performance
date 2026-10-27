from typing import Any, cast

from aisc_plugin_interface import (
    BaseEvaluationPlugin,
    InputType,
    evaluation_input,
)

from .config_form import ConfigForm
from .data_input_provider import DataFrameProvider
from .model_input_provider import OnnxInputProvider


@evaluation_input(
    name="datashape",
    label="Data Shape",
    input_type=InputType.DATASHAPE,
    required=False,
)
@evaluation_input(
    name="test-dataset",
    label="Test Dataset",
    input_provider_class=DataFrameProvider,
    input_type=InputType.DATASET,
)
@evaluation_input(
    name="model",
    label="Model",
    input_provider_class=OnnxInputProvider,
    input_type=InputType.MODEL,
)
class BasePerformanceEvaluationPlugin(BaseEvaluationPlugin[ConfigForm]):
    """Base class for performance evaluation plugins."""

    @staticmethod
    def _datashape_features(payload: object) -> list[dict[str, Any]]:
        """Normalize a datashape payload to a list of feature dicts.

        Accepts a full ``DataShape`` payload and the default features-only
        payload the platform emits (``{"features": [...]}``).
        """
        if not isinstance(payload, dict):
            return []
        raw_features = payload.get("features")
        if not isinstance(raw_features, list):
            return []
        return [cast("dict[str, Any]", f) for f in raw_features if isinstance(f, dict)]

    def _read_datashape(self) -> tuple[str, str | None, list[str]]:
        """Resolve (target, date, feature columns) from the datashape input.

        The datashape component's ``json_value`` is the feature schema: a full
        ``DataShape`` payload or the default features-only payload
        (``{"features": [...]}``).  Feature ``role`` designates the target, the
        optional date axis, and the model features.

        Returns:
            Tuple of (target column, date column or None, feature columns).

        Raises:
            ValueError: If the datashape input is missing or has no target.
        """
        payload = self.get_input_data("datashape")
        if payload is None:
            raise ValueError(
                "The Data Shape input is required for evaluation. "
                "Create a data shape from the uploaded dataset on the project "
                "Settings page and attach it as an evaluation input."
            )

        features = self._datashape_features(payload)
        if not features:
            raise ValueError(
                "The Data Shape input carries no features; a data shape must be "
                "derived from a dataset on the project Settings page."
            )

        target_col = next(
            (str(f.get("name")) for f in features if f.get("role") == "target"),
            None,
        )
        if target_col is None:
            raise ValueError(
                "The Data Shape must designate a target feature (a feature with role 'target')."
            )

        date_feature = next((str(f.get("name")) for f in features if f.get("role") == "date"), None)
        columns_features = [str(f.get("name")) for f in features if f.get("role") == "feature"]

        return target_col, date_feature, columns_features
