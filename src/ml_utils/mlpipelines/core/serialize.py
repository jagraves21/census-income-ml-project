from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def _serialize(node, node_name=None):
	node_name = node.__class__.__name__ if node_name is None else node_name

	if isinstance(node, Pipeline):
		params = node.get_params(deep=False)

		return {
			"name": node_name,
			"params": {
				key: value
				for key, value in params.items()
				if key != "steps"
			},
			"steps": [
				_serialize(step, step_name)
				for step_name, step in node.steps
			],
		}

	if isinstance(node, ColumnTransformer):
		params = node.get_params(deep=False)

		return {
			"name": node_name,
			"params": {
				key: value
				for key, value in params.items()
				if key != "transformers"
			},
			"transformers": [
				{
					"name": transformer_name,
					"columns": columns,
					"transformer": _serialize(transformer),
				}
				for transformer_name, transformer, columns in node.transformers
			],
		}

	params = node.get_params(deep=False)

	return {
		"name": node.__class__.__name__,
		"params": params,
	}


def serialize_pipeline(obj):
	return _serialize(obj)

