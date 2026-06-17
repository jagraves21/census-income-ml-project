import warnings
from sklearn.utils.discovery import all_estimators

from ..registry import (
	make_builder,
	_register_estimator
)


def load_sklearn_estimators():
	for type_filter in {"classifier", "regressor", "cluster", "transformer"}:
		for name, cls in all_estimators(type_filter=type_filter):
			try:
				builder = make_builder(name, cls)

				_register_estimator(
					name=name,
					builder=make_builder(cls),
					cls=cls,
					kind=type_filter
				)

			except ValueError:
				continue
			except Exception as e:
				warnings.warn(
					f"Failed to register sklearn estimator '{name}': "
					f"{type(e).__name__}: {e}",
					RuntimeWarning,
					stacklevel=2,
				)

