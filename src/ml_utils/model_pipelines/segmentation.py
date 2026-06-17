from copy import deepcopy

from ..mlpipelines import build_pipeline

from .utils import _get_available_models
from .preprocessing import _get_preprocessing_spec
from .dimensionality_reduction import _get_dimensionality_reduction_spec


from ..utils import(
	is_string,
	is_sequence
)


_CLUSTERING_SPECS = {
	"kmeans": {
		"name": "KMeans"
	},
	"dbscan": {
		"name": "DBSCAN"
	},
	"hdbscan": {
		"name": "HDBSCAN",
		"kind": "third_party",
		"params": {
			"min_cluster_size": 800,
			"min_samples": 50,
			"cluster_selection_method": "eom"
			#"min_cluster_size": 500,
			#"min_samples": 50,
			#"cluster_selection_method": "eom"
		}
	},
}


_CLUSTERING_FAMILIES = {
	"centroid": ["kmeans"],
	"density": ["dbscan", "hdbscan"],
}


def get_available_segmentation_models():
	return _get_available_models(
		_CLUSTERING_SPECS,
		_CLUSTERING_FAMILIES,
	)


def _build_base_segmentation_spec(categorical_columns, numeric_columns, remove_niu):
	return _get_preprocessing_spec(
		categorical_columns=categorical_columns,
		numeric_columns=numeric_columns,
		remove_niu=remove_niu,
	)


def _get_segmentation_spec(model, **params):
	spec = deepcopy(_CLUSTERING_SPECS[model])

	if len(params) > 0:
		spec.setdefault("params", {}).update(params)

	return spec


def get_segmentation_models(
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
			"params": {"verbose": True},
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
				_get_segmentation_spec(model_name),
			]
		}
		return build_pipeline(spec)

	# all models
	if select is None:
		return {name: build(name) for name in _CLUSTERING_SPECS}

	# family
	if is_string(select) and select in _CLUSTERING_FAMILIES:
		return {
			name: build(name)
			for name in _CLUSTERING_FAMILIES[select]
		}

	# single model
	if is_string(select):
		if select not in _CLUSTERING_SPECS:
			raise KeyError(
				f"Unknown model '{select}'. "
				f"Valid models: {list(_CLUSTERING_SPECS)}, "
				f"families: {list(_CLUSTERING_FAMILIES)}"
			)
		return build(select)

	# list of models
	if is_sequence(select):
		invalid = [name for name in select if name not in _CLUSTERING_SPECS]
		if invalid:
			raise KeyError(
				f"Unknown models: {invalid}. "
				f"Valid models: {list(_CLUSTERING_SPECS)}"
			)

		return {name: build(name) for name in select}

	raise TypeError(
		"select must be None, str, or iterable of str"
	)

