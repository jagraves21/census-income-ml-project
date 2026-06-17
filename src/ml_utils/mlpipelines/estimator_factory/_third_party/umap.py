import warnings

from ..registry import (
	make_builder,
	_register_estimator,
)

def register_umap():
	try:
		from umap import UMAP
		
		_register_estimator(
			name="UMAP",
			builder=make_builder(UMAP),
			cls=UMAP,
			kind="third_party",
		)
	except ImportError:
		warnings.warn("umap not installed")

