import pandas as pd


from .utils import display_importance_summary


def importance(
	pipeline,
	preprocessor_step="preprocessing",
	model_step="DecisionTreeClassifier",
	type_separator="__",
	value_separator="_"
):
	preprocessor = pipeline[preprocessor_step]
	model = pipeline[model_step]

	raw_features = preprocessor.get_feature_names_out()
	importances = model.feature_importances_

	feature_types = []
	feature_names = []
	feature_values = []

	for feature in raw_features:
		type_part, rest = feature.split(type_separator, 1)

		name_parts = rest.split(value_separator, 1)
		name = name_parts[0]
		value = name_parts[1] if len(name_parts) > 1 else None

		feature_types.append(type_part)
		feature_names.append(name)
		feature_values.append(value)

	importance_df = pd.DataFrame({
		"feature_raw": raw_features,
		"feature_type": feature_types,
		"feature_name": feature_names,
		"feature_value": feature_values,
		"gini_importance": importances
	})

	return importance_df.sort_values(
		"gini_importance",
		ascending=False
	).reset_index(drop=True)


def display_decision_tree_importance(
	df,
	top_n=10
):
	return display_importance_summary(
		df,
		importance_column="gini_importance",
		sign_column=None,
		top_n=top_n,
		split_positive_negative=False
	)

