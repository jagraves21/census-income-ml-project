from . import classifiers
from .classifiers import XGBStringClassifier

from . import transformers
from .transformers import (
	OneHotEncoderWithIgnore,
	ValueReplacer
)


__all__ = [
	"XGBStringClassifier",

	"OneHotEncoderWithIgnore",
	"ValueReplacer"
]


_loaded = False

def _ensure_loaded():
	global _loaded
	if _loaded:
		return

	from ._register import register_estimators
	register_estimators()

	_loaded = True


_ensure_loaded()

