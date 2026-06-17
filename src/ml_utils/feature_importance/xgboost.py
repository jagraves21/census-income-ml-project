import numpy as np
import pandas as pd
import shap

from .utils import display_importance_summary

from ..utils import is_list


def importance(
	pipeline,
	X,
	preprocessor_step="preprocessing",
	model_step="XGBStringClassifier",
	type_separator="__",
	value_separator="_"
):
	preprocessor = pipeline[preprocessor_step]
	model = pipeline[model_step].model_

	X_transformed = preprocessor.transform(X)
	feature_names = preprocessor.get_feature_names_out()

	explainer = shap.TreeExplainer(model)
	shap_values = explainer.shap_values(X_transformed)

	if is_list(shap_values):
		shap_values = shap_values[1]

	feature_types = []
	feature_names_clean = []
	feature_values = []

	for feature in feature_names:
		type_part, rest = feature.split(type_separator, 1)

		name_parts = rest.split(value_separator, 1)
		name = name_parts[0]
		value = name_parts[1] if len(name_parts) > 1 else None

		feature_types.append(type_part)
		feature_names_clean.append(name)
		feature_values.append(value)

	df = pd.DataFrame({
		"feature_raw": feature_names,
		"feature_type": feature_types,
		"feature_name": feature_names_clean,
		"feature_value": feature_values,
		"mean_abs_shap": np.abs(shap_values).mean(axis=0),
		"mean_shap": shap_values.mean(axis=0),
		"positive_ratio": np.mean(shap_values > 0, axis=0)
	})

	df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

	return df


def display_xgboost_importance(
	df,
	top_n=10
):
	return display_importance_summary(
		df,
		importance_column="mean_abs_shap",
		sign_column=None,
		top_n=top_n,
		split_positive_negative=False
	)

