from ...mlpipelines import register_estimator

from .classifiers import XGBStringClassifier

from .transformers import (
	OneHotEncoderWithIgnore,
	ValueReplacer
)


_ESTIMATORS = {
	"XGBStringClassifier": XGBStringClassifier,
	"OneHotEncoderWithIgnore": OneHotEncoderWithIgnore,
	"ValueReplacer": ValueReplacer
}


def register_estimators():
	for name,cls in _ESTIMATORS.items():
		register_estimator(name=name, cls=cls)

