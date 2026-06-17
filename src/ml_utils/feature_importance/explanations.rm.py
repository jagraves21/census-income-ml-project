import numpy as np
import pandas as pd
from IPython.display import HTML, display


def extract_logistic_regression_importance(
	pipeline,
	preprocessor_step="preprocessing",
	model_step="model",
	type_separator="__",
	value_separator="_"
):
	preprocessor = pipeline[preprocessor_step]
	model = pipeline[model_step]

	raw_features = preprocessor.get_feature_names_out()
	coefficients = model.coef_.ravel()

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

	coef_df = pd.DataFrame({
		"feature_raw": raw_features,
		"feature_type": feature_types,
		"feature_name": feature_names,
		"feature_value": feature_values,
		"coefficient": coefficients
	})
	coef_df["abs_coefficient"] = np.abs(coef_df["coefficient"])
	coef_df["odds_ratio"] = np.exp(coef_df["coefficient"])
	coef_df = coef_df.sort_values("abs_coefficient", ascending=False)
	coef_df = coef_df.reset_index(drop=True)

	return coef_df


def display_logistic_regression_summary(coef_df, top_n=None):
	if top_n is None:
		pos_df = coef_df.sort_values("coefficient", ascending=False)
		neg_df = coef_df.sort_values("coefficient", ascending=True)
	else:
		pos_df = coef_df.sort_values("coefficient", ascending=False).head(top_n)
		neg_df = coef_df.sort_values("coefficient", ascending=True).head(top_n)

	display(HTML("<h4>Top Positive Drivers</h4>"))
	display(pos_df)

	display(HTML("<h4>Top Negative Drivers</h4>"))
	display(neg_df)

	group_type = (
		coef_df
		.groupby("feature_type", as_index=False)["abs_coefficient"]
		.sum()
		.rename(columns={
			"feature_type": "feature_group",
			"abs_coefficient": "importance"
		})
		.sort_values("importance", ascending=False)
	)

	display(HTML("<h4>Feature Importance by Type</h4>"))
	display(group_type)

	group_type_name = (
		coef_df
		.groupby(
			["feature_type", "feature_name"], as_index=False
		)["abs_coefficient"]
		.sum()
		.rename(columns={
			"abs_coefficient": "importance"
		})
		.sort_values("importance", ascending=False)
	)

	display(HTML("<h4>Feature Importance by Type + Name</h4>"))
	display(group_type_name)


def extract_decision_tree_importance(
	pipeline,
	preprocessor_step="preprocessing",
	model_step="model",
	type_separator="__",
	value_separator="_"
):
	preprocessor = pipeline.named_steps[preprocessor_step]
	model = pipeline.named_steps[model_step]

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
	importance_df = importance_df.sort_values(
		"gini_importance", ascending=False
	)
	importance_df = importance_df.reset_index(drop=True)

	return importance_df


def display_decision_tree_summary(importance_df, top_n=None):
	if top_n is None:
		top_df = importance_df.sort_values("gini_importance", ascending=False)
	else:
		top_df = importance_df.sort_values(
			"gini_importance", ascending=False
		).head(top_n)

	display(HTML("<h4>Top Important Features (Gini Importance)</h4>"))
	display(top_df)

	group_type = (
		importance_df
		.groupby("feature_type", as_index=False)["gini_importance"]
		.sum()
		.rename(columns={
			"feature_type": "feature_group",
			"gini_importance": "importance"
		})
		.sort_values("importance", ascending=False)
	)

	display(HTML("<h4>Feature Importance by Type</h4>"))
	display(group_type)

	group_type_name = (
		importance_df
		.groupby(
			["feature_type", "feature_name"], as_index=False
		)["gini_importance"]
		.sum()
		.rename(columns={
			"gini_importance": "importance"
		})
		.sort_values("importance", ascending=False)
	)

	display(HTML("<h4>Feature Importance by Type + Name</h4>"))
	display(group_type_name)

