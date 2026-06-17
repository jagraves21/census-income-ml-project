import warnings

from ..registry import (
	make_builder,
	_register_estimator,
)

def register_xgboost():
	try:
		from xgboost import XGBClassifier
		
		_register_estimator(
			name="XGBClassifier",
			builder=make_builder(XGBClassifier),
			cls=XGBClassifier,
			kind="third_party",
		)
	except ImportError:
		warnings.warn("xgboost not installed")

