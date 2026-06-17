from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from ..schemas import validate_pipeline
from ..estimator_factory import build_estimator


def _is_pipeline(node):
	return "steps" in node


def _is_column_transformer(node):
	return "transformers" in node


def _is_estimator(node):
	return not (
		_is_pipeline(node) or _is_column_transformer(node)
	)


def _derive_step_name(node):
	return node["name"]


def _make_estimator_object(
	name,
	kind,
	**params,
):
	return build_estimator(name, kind=kind, **params)


def _compile_estimator(node):
	name = node["name"]
	kind = node.get("kind", None)
	params = node.get("params", {})

	return _make_estimator_object(
		name=name,
		kind=kind,
		**params,
	)


def _make_column_transformer_object(transformers, params):
	return ColumnTransformer(
		transformers=transformers,
		**params
	)


def _compile_column_transformer(node):
	name = node["name"]
	params = node.get("params", {})
	transformers = node["transformers"]

	compiled_transformers = []

	for transformer_spec in transformers:
		transformer_name = transformer_spec["name"]
		columns = transformer_spec["columns"]

		transformer = _compile_node(
			transformer_spec["transformer"]
		)

		compiled_transformers.append(
			(transformer_name, transformer, columns)
		)

	return _make_column_transformer_object(
		transformers=compiled_transformers,
		params=params,
	)


def _make_pipeline_object(steps, params):
	return Pipeline(
		steps=steps,
		**params
	)


def _compile_pipeline(node):
	name = node["name"]
	params = node.get("params", {})
	steps = node["steps"]

	compiled_steps = []

	for step in steps:
		compiled_step = _compile_node(step)
		step_name = _derive_step_name(step)

		compiled_steps.append(
			(step_name, compiled_step)
		)

	return _make_pipeline_object(
		steps=compiled_steps,
		params=params,
	)


def _compile_node(node):
	if _is_estimator(node):
		return _compile_estimator(node)

	if _is_column_transformer(node):
		return _compile_column_transformer(node)
		
	if _is_pipeline(node):
		return _compile_pipeline(node)

	raise ValueError(f"Unknown node type: {node}")


def build_pipeline(spec):
	validate_pipeline(spec)
	return _compile_node(spec)

