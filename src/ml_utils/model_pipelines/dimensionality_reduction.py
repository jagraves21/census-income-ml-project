from copy import deepcopy

from ..mlpipelines import build_pipeline
from .utils import _get_available_models
from .preprocessing import _get_preprocessing_spec

from ..utils import(
	is_string,
	is_sequence
)


_DECOMPOSITION_SPEC = {
	"pca": {
		"name": "PCA"
	},
	"svd": {
		"name": "TruncatedSVD"
	},
	"umap": {
		"name": "UMAP",
		"params": {
			"n_components": 20,
			"n_neighbors": 50,
			"min_dist": 0.0,
			"metric": "cosine",
			"init": "random",
			"random_state": 42,
			"low_memory": True,
			"verbose": True,

			#"n_components": 20,
			#"n_neighbors": 30,
			#"min_dist": 0.0,
			#"metric": "euclidean",
			#"random_state": 42
		}
	},
}


_DECOMPOSITION__FAMILIES = {
	"linear": ["pca", "svd"],
	"manifold": ["umap"],
}


def get_available_dimensionality_reduction_models():
	return _get_available_models(
		_DECOMPOSITION_SPEC,
		_DECOMPOSITION__FAMILIES,
	)


def _get_dimensionality_reduction_spec(model, **params):
	spec = deepcopy(_DECOMPOSITION_SPEC[model])

	if len(params) > 0:
		spec.setdefault("params", {}).update(params)

	return spec


def get_dimensionality_reduction_models(
	categorical_columns,
	numeric_columns,
	remove_niu=None,
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
					remove_niu=remove_niu
				),
				_get_dimensionality_reduction_spec(
					model_name, n_components=n_components
				)
			]
		}
		return build_pipeline(spec)

	# all models
	if select is None:
		return {name: build(name) for name in _DECOMPOSITION_SPEC}

	# family
	if is_string(select) and select in _DECOMPOSITION__FAMILIES:
		return {
			name: build(name)
			for name in _DECOMPOSITION__FAMILIES[select]
		}

	# single model
	if is_string(select):
		if select not in _DECOMPOSITION_SPEC:
			raise KeyError(
				f"Unknown model '{select}'. "
				f"Valid models: {list(_DECOMPOSITION_SPEC)}, "
				f"families: {list(_DECOMPOSITION__FAMILIES)}"
			)
		return build(select)

	# list of models
	if is_sequence(select):
		invalid = [
			name for name in select
			if name not in _DECOMPOSITION_SPEC
		]
		if invalid:
			raise KeyError(
				f"Unknown models: {invalid}. "
				f"Valid models: {list(_DECOMPOSITION_SPEC)}"
			)

		return {name: build(name) for name in select}

	raise TypeError(
		"select must be None, str, or iterable of str"
	)

