from copy import deepcopy

from ..mlpipelines import build_pipeline

from .preprocessing import _get_preprocessing_spec
from .dimensionality_reduction import _get_dimensionality_reduction_spec

from ..utils import(
	is_string,
	is_sequence
)


_MODEL_SPECS = {
	# "logistic_regression": {
	#	"name": "LogisticRegression",
	#	"params": {
	#		# "penalty": "none",
	#		"C": np.inf,
	#		"solver": "lbfgs",
	#		#"max_iter": 500,
	#		"random_state": 42
	#	}
	#},
	"lasso_logistic_regression": {
		"name": "LogisticRegression",
		"params": {
			#"penalty": "l1",
			"l1_ratio": 1,
			"solver": "saga",
			#"max_iter": 1000,
			"random_state": 42
		}
	},
	"ridge_logistic_regression": {
		"name": "LogisticRegression",
		"params": {
			#"penalty": "l2",
			"l1_ratio": 0,
			"solver": "lbfgs",
			#"max_iter": 1000,
			"random_state": 42
		}
	},
	"decision_tree": {
		"name": "DecisionTreeClassifier",
		"params": {
			"criterion": "gini",
			"random_state": 42
		}
	},
	"xgboost": {
		"name": "XGBStringClassifier",
		"params": {
			"n_estimators": 300,
			"max_depth": 4,
			"learning_rate": 0.05,
			"subsample": 0.8,
			"colsample_bytree": 0.8,
			"random_state": 42,
			"eval_metric": "logloss",
			"n_jobs": -1,
			"enable_categorical": False,
		}
	}
}


_MODEL_FAMILIES = {
	"linear": ["ridge_logistic_regression", "lasso_logistic_regression"],
	"trees": ["decision_tree"],
	#"boosting": ["xgboost"]
}


def get_available_prediction_models():
	return _get_available_models(
		_MODEL_SPECS,
		_MODEL_FAMILIES,
	)


def _get_prediction_spec(model, **params):
	spec = deepcopy(_MODEL_SPECS[model])

	if len(params) > 0:
		spec.setdefault("params", {}).update(params)

	return spec


def get_prediction_models(
	categorical_columns,
	numeric_columns,
	remove_niu=True,
	dimensionality_reduction=None,
	n_components=None,
	select=None
):
	def build(model_name):
		spec = {
			"name": "Pipeline",
			"steps": [
				_get_preprocessing_spec(
					categorical_columns=categorical_columns,
					numeric_columns=numeric_columns,
					remove_niu=remove_niu,
					),
				*(
					[
						_get_dimensionality_reduction_spec(
							dimensionality_reduction,
							n_components=n_components,
						)
					]
					if dimensionality_reduction is not None
					else []
				),
				_get_prediction_spec(model_name),
			]
		}
		return build_pipeline(spec)

	# all models
	if select is None:
		return {name: build(name) for name in _MODEL_SPECS}

	# family
	if is_string(select) and select in _MODEL_FAMILIES:
		return {
			name: build(name)
			for name in _MODEL_FAMILIES[select]
		}

	# single model
	if is_string(select):
		if select not in _MODEL_SPECS:
			raise KeyError(
				f"Unknown model '{select}'. "
				f"Valid models: {list(_MODEL_SPECS)}, "
				f"families: {list(_MODEL_FAMILIES)}"
			)
		return build(select)

	# list of models
	if is_sequence(select):
		invalid = [name for name in select if name not in _MODEL_SPECS]
		if invalid:
			raise KeyError(
				f"Unknown models: {invalid}. "
				f"Valid models: {list(_MODEL_SPECS)}"
			)

		return {name: build(name) for name in select}

	raise TypeError(
		"select must be None, str, or iterable of str"
	)

