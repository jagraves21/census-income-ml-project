import warnings

from ..registry import (
	make_builder,
	_register_estimator,
)

def register_hdbscan():
	try:
		from hdbscan import HDBSCAN
		
		_register_estimator(
			name="HDBSCAN",
			builder=make_builder(HDBSCAN),
			cls=HDBSCAN,
			kind="third_party",
		)
	except ImportError:
		warnings.warn("hdbscan not installed")

